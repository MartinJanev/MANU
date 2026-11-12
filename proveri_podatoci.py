#!/usr/bin/env python3
# =====================================================================
# МАНУ: проверка на симнатите податоци (IXI / MOLAB / open_ms_data)
#
# Користи само стандардна Python библиотека, па работи и на најавниот
# јазол без контејнер:
#   python3 proveri_podatoci.py                         # брза проверка
#   python3 proveri_podatoci.py --full                  # + целосна проверка на .gz (неколку минути)
#   python3 proveri_podatoci.py --rezultati ~/manu/results/medgemma
#   python3 proveri_podatoci.py --ms-oznaki ~/manu/data/clinical_data.csv
#
# Што проверува (наспроти изворот, не наспроти бројките од трудот):
#   1. IXI: дали е распакувано сè од IXI-T1.tar (листа IXI-T1.list, ја прави
#      prezemi_podatoci.sh), со точни големини
#   2. MOLAB: дали се симнати сите датотеки од figshare колекцијата
#      (manifest.tsv: md5 за xlsx, листа на секоја zip архива) и дали
#      molab/nifti/ (што ја користи run_radfm.py) е иста со figshare сликите
#   3. open_ms_data: дали сите датотеки од git репото се на диск (git ls-files)
#   4. Дали секоја датотека е читлив 3D NIfTI (не е празна, скратена,
#      git-lfs покажувач или 4D)
#   5. MS ознаки: дистрибуција во open_ms_data/patient_info.csv
#   6. (опционално) дали истите датотеки се во излезите на другите модели
# Излезен код: 0 ако нема грешки, 1 ако има.
# =====================================================================
import argparse
import collections
import csv
import filecmp
import gzip
import hashlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

GRESKI = []      # критични проблеми
VNIMANIE = []    # работи за проверка


def ok(msg):
    print(f"  [OK]        {msg}")


def greska(msg):
    GRESKI.append(msg)
    print(f"  [ГРЕШКА]    {msg}")


def vnimanie(msg):
    VNIMANIE.append(msg)
    print(f"  [ВНИМАНИЕ]  {msg}")


def info(msg):
    print(f"              {msg}")


def naslov(t):
    print(f"\n=== {t} " + "=" * max(0, 60 - len(t)))


def stem(p):
    n = Path(p).name
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


# ---------------------------------------------------------------------
# NIfTI заглавие и интегритет (без nibabel)
# ---------------------------------------------------------------------
def procitaj_zaglavie(raw):
    """Враќа dict со dims, pixdim, bitpix, vox_offset или None."""
    if len(raw) < 348:
        return None
    for end in ("<", ">"):
        size = struct.unpack(end + "i", raw[:4])[0]
        if size == 348:
            dim = struct.unpack(end + "8h", raw[40:56])
            bitpix = struct.unpack(end + "h", raw[72:74])[0]
            pixdim = struct.unpack(end + "8f", raw[76:108])
            vox_offset = struct.unpack(end + "f", raw[108:112])[0]
            return dict(dim=dim, bitpix=bitpix, pixdim=pixdim, vox_offset=int(vox_offset))
        if size == 540 and len(raw) >= 540:
            bitpix = struct.unpack(end + "h", raw[14:16])[0]
            dim = struct.unpack(end + "8q", raw[16:80])
            pixdim = struct.unpack(end + "8d", raw[104:168])
            vox_offset = struct.unpack(end + "q", raw[168:176])[0]
            return dict(dim=dim, bitpix=bitpix, pixdim=pixdim, vox_offset=int(vox_offset))
    return None


def ocekuvana_golemina(h):
    n = max(1, min(int(h["dim"][0]), 7))
    vox = 1
    for d in h["dim"][1 : n + 1]:
        vox *= max(1, int(d))
    return h["vox_offset"] + vox * h["bitpix"] // 8


def proveri_nifti(path, full=False):
    """Враќа (problem или None, предупредување или None, заглавие или None)."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"не може да се прочита ({e})", None, None
    if size == 0:
        return "празна датотека (0 B)", None, None
    with open(p, "rb") as f:
        pocetok = f.read(200)
    if pocetok.startswith(b"version https://git-lfs"):
        return "git-lfs покажувач, не вистинска слика", None, None

    gz = p.name.endswith(".gz")
    try:
        if gz:
            with gzip.open(p, "rb") as f:
                raw = f.read(544)
        else:
            with open(p, "rb") as f:
                raw = f.read(544)
    except (OSError, EOFError) as e:
        return f"оштетена датотека ({e.__class__.__name__})", None, None

    h = procitaj_zaglavie(raw)
    if h is None:
        return "не е валидно NIfTI заглавие", None, None

    dims = [int(d) for d in h["dim"][1 : 1 + max(1, min(int(h["dim"][0]), 7))]]
    if int(h["dim"][0]) < 3 or any(d <= 1 for d in dims[:3]):
        return f"не е 3D волумен (dim={dims})", None, h
    upoz = None
    if len(dims) >= 4 and dims[3] > 1:
        upoz = f"4D волумен {dims} (моделите очекуваат 3D)"

    ocek = ocekuvana_golemina(h)
    if not gz:
        if size < ocek:
            return f"скратена: {size} B, очекувано ≥ {ocek} B", None, h
    elif full:
        n = 0
        try:
            with gzip.open(p, "rb") as f:
                while True:
                    b = f.read(1 << 22)
                    if not b:
                        break
                    n += len(b)
        except (OSError, EOFError) as e:
            return f"оштетен .gz ({e.__class__.__name__}: {e})", None, h
        if n < ocek:
            return f"скратена: {n} B откомпресирано, очекувано ≥ {ocek} B", None, h
    else:
        # Брза проверка: последните 4 бајти на .gz = откомпресирана големина (mod 2^32)
        with open(p, "rb") as f:
            f.seek(-4, os.SEEK_END)
            isize = struct.unpack("<I", f.read(4))[0]
        # мала разлика (до 1 MB) = дополнителни бајти на крајот, не е проблем
        if (isize - ocek) % (1 << 32) >= (1 << 20):
            w = "можеби скратена/оштетена (крајот на .gz не одговара) — пуштете со --full"
            upoz = f"{upoz}; {w}" if upoz else w
    return None, upoz, h


def oblik(h):
    n = max(1, min(int(h["dim"][0]), 7))
    return "x".join(str(int(d)) for d in h["dim"][1 : n + 1])


def proveri_mnozestvo(ime, pateki, full):
    """Проверува интегритет; враќа dict pateka -> заглавие."""
    zaglavija, problemi, upozoruvanja = {}, [], []
    oblici = collections.Counter()
    for pth in pateki:
        prob, upoz, h = proveri_nifti(pth, full)
        if prob:
            problemi.append((pth, prob))
        if upoz:
            upozoruvanja.append((pth, upoz))
        if h is not None:
            zaglavija[pth] = h
            oblici[oblik(h)] += 1
    if problemi:
        greska(f"{ime}: {len(problemi)} неисправни датотеки")
        for pth, prob in problemi[:15]:
            info(f"- {pth}: {prob}")
        if len(problemi) > 15:
            info(f"  ... и уште {len(problemi) - 15}")
    else:
        ok(f"{ime}: сите {len(pateki)} датотеки се читливи 3D NIfTI"
           + (" (целосна .gz проверка)" if full else " (брза проверка)"))
    if upozoruvanja:
        vnimanie(f"{ime}: {len(upozoruvanja)} датотеки со предупредување")
        for pth, u in upozoruvanja[:10]:
            info(f"- {pth}: {u}")
    if oblici:
        info("Најчести димензии: " + ", ".join(f"{o} ({n})" for o, n in oblici.most_common(5))
             + (f" ... ({len(oblici)} различни)" if len(oblici) > 5 else ""))
    return zaglavija


def najdi(koren, pattern_re):
    koren = Path(koren)
    if not koren.is_dir():
        return []
    rx = re.compile(pattern_re)
    return sorted(str(p) for p in koren.rglob("*") if p.is_file() and rx.search(p.name))


def proveri_lista(ime, koren, lista):
    """Дали секоја датотека од листата на архивата (патека<TAB>големина) е под koren
    со иста големина. Враќа листа од пронајдените патеки."""
    najdeni, nema, golemina = [], [], []
    for red in Path(lista).read_text(encoding="utf-8").splitlines():
        rel, gol = red.rsplit("\t", 1)
        p = Path(koren) / rel
        if not p.is_file():
            nema.append(rel)
        elif p.stat().st_size != int(gol):
            golemina.append(f"{rel} ({p.stat().st_size} B, во архивата {gol} B)")
        else:
            najdeni.append(str(p))
    vkupno = len(najdeni) + len(nema) + len(golemina)
    if nema or golemina:
        greska(f"{ime}: {len(najdeni)} / {vkupno} датотеки од архивата се на диск")
        for x in (nema + golemina)[:10]:
            info(f"- {'нема ' if x in nema else 'друга големина: '}{x}")
    else:
        ok(f"{ime}: сите {vkupno} датотеки од архивата се на диск (иста големина)")
    return najdeni


# ---------------------------------------------------------------------
# 1. IXI
# ---------------------------------------------------------------------
def proveri_ixi(data, full):
    naslov("IXI (здрави контроли, T1)")
    lista = data / "ixi" / "IXI-T1.list"
    if not lista.is_file():
        greska(f"нема {lista}: пуштете prezemi_podatoci.sh (ја симнува IXI-T1.tar и ја запишува листата)")
        fajlovi = najdi(data / "ixi" / "T1", r"-T1\.nii(\.gz)?$")
    else:
        fajlovi = proveri_lista("IXI-T1.tar", data / "ixi" / "T1", lista)
    if not fajlovi:
        return {}
    proveri_mnozestvo("IXI", fajlovi, full)
    return {stem(p): p for p in fajlovi}


# ---------------------------------------------------------------------
# 2. MOLAB
# ---------------------------------------------------------------------
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def figshare_ime(ime):
    """molab/nifti име -> figshare име: 010020N__010020_19010822_X_img.nii -> 10020_19010822_X_img.nii"""
    m = re.match(r"\d+N__0*(\d+_.+)$", ime)
    return m.group(1) if m else ime


def proveri_molab(data, full):
    naslov("MOLAB (метастази, figshare 10.6084/m9.figshare.c.6194104)")
    fs = data / "molab" / "figshare"
    manifest = fs / "manifest.tsv"
    if not manifest.is_file():
        greska(f"нема {manifest}: пуштете prezemi_podatoci.sh (ја симнува целата figshare колекција)")
    else:
        redovi = [r.split("\t") for r in manifest.read_text(encoding="utf-8").splitlines() if r]
        info(f"figshare колекцијата има {len(redovi)} датотеки")
        for ime, gol, hes, _ in redovi:
            if ime.endswith(".zip"):
                lista = fs / f"{ime}.list"
                if lista.is_file():
                    proveri_lista(ime, fs, lista)
                else:
                    greska(f"{ime}: не е симнат/распакуван (нема {lista.name})")
            elif not (fs / ime).is_file():
                greska(f"{ime}: не е симнат")
            elif md5(fs / ime) != hes:
                greska(f"{ime}: md5 не одговара на figshare")
            else:
                ok(f"{ime}: md5 одговара на figshare")

    izvor = {Path(p).name: p for p in najdi(fs / "NIFTIs_Images", r"_img\.nii(\.gz)?$")}
    if izvor:
        pac = {re.match(r"(\d+)_", n).group(1) for n in izvor if re.match(r"(\d+)_", n)}
        info(f"figshare NIfTI слики: {len(izvor)} кај {len(pac)} пациенти")
        proveri_mnozestvo("MOLAB figshare слики", list(izvor.values()), full)

    # molab/nifti/ е рачно качена верзија со други имиња; ја користи run_radfm.py
    nasi = {figshare_ime(Path(p).name): p for p in najdi(data / "molab" / "nifti", r"_img\.nii(\.gz)?$")}
    if not nasi:
        info("нема molab/nifti/*_img.nii (run_radfm.py ги чита од таму)")
        return {}
    if izvor:
        nema = sorted(set(izvor) - set(nasi))
        visok = sorted(set(nasi) - set(izvor))
        razl = [n for n in sorted(set(izvor) & set(nasi))
                if (not filecmp.cmp(izvor[n], nasi[n], shallow=False) if full
                    else Path(izvor[n]).stat().st_size != Path(nasi[n]).stat().st_size)]
        if nema or visok or razl:
            greska(f"molab/nifti/ ({len(nasi)}) не е иста со figshare ({len(izvor)}): "
                   f"недостасуваат {len(nema)}, вишок {len(visok)}, различни {len(razl)}")
            for n in (nema[:5] + visok[:5] + razl[:5]):
                info(f"- {n}")
        else:
            ok(f"molab/nifti/: сите {len(nasi)} слики се исти со figshare "
               + ("(бајт по бајт)" if full else "(иста големина; --full споредува бајт по бајт)"))
    else:
        proveri_mnozestvo("MOLAB molab/nifti", list(nasi.values()), full)
    return {stem(p): p for p in nasi.values()}


# ---------------------------------------------------------------------
# 3. open_ms_data
# ---------------------------------------------------------------------
MS_VERZII = {
    "raw": ("cross_sectional/raw", r"^(patient\d{2}_)?T1W?\.nii(\.gz)?$"),
    "coregistered": ("cross_sectional/coregistered", r"^(patient\d{2}_)?T1W\.nii(\.gz)?$"),
    "coregistered_resampled": ("cross_sectional/coregistered_resampled", r"^T1W\.nii(\.gz)?$"),
    "MNI (skull-stripped)": ("cross_sectional/MNI", r"^T1_N4_.*\.nii(\.gz)?$"),
}


def pac_id(pateka):
    m = re.search(r"patient(\d{2})(?!\d)", str(pateka))
    return m.group(1) if m else None


def proveri_ms(data, full, ms_oznaki):
    naslov("open_ms_data (мултиплекс склероза, T1)")
    koren = data / "open_ms_data"
    if not koren.is_dir():
        greska(f"нема папка {koren}")
        return {}

    try:
        git = subprocess.run(["git", "-C", str(koren), "ls-files", "-z"],
                             capture_output=True, check=True).stdout.decode().split("\0")
        git = [f for f in git if f]
        nema = [f for f in git if not (koren / f).exists()]
        if nema:
            greska(f"open_ms_data: {len(git) - len(nema)} / {len(git)} датотеки од git репото се на диск")
            for f in nema[:10]:
                info(f"- нема {f}")
        else:
            ok(f"open_ms_data: сите {len(git)} датотеки од git репото се на диск")
    except (OSError, subprocess.CalledProcessError) as e:
        vnimanie(f"не можам да го прочитам git репото ({e}) — пуштете prezemi_podatoci.sh")

    verzii = {}
    for ime, (pod, rx) in MS_VERZII.items():
        fajlovi = najdi(koren / pod, rx)
        verzii[ime] = {pac_id(p): p for p in fajlovi if pac_id(p)}
    raw = verzii["raw"]
    info(f"{len(raw)} пациенти со T1 во cross_sectional/raw")
    info("T1 верзии во репото: " + ", ".join(f"{k} ({len(v)})" for k, v in verzii.items()))
    long_n = len(najdi(koren / "longitudinal", r"\.nii(\.gz)?$"))
    if long_n:
        info(f"longitudinal/: {long_n} NIfTI (за промени во лезии) — не се користи")

    proveri_mnozestvo("MS T1 (raw)", list(raw.values()), full)

    # Кои patientNN_T1 датотеки ги користевте и од која верзија се?
    rasprseni = [p for p in najdi(data, r"^patient\d{2}_T1\.nii(\.gz)?$")]
    if rasprseni:
        info(f"Најдени {len(rasprseni)} датотеки patientNN_T1.nii* (како во трудот). Од која верзија се?")
        sovpaganje = collections.Counter()
        for p in rasprseni:
            _, _, h = proveri_nifti(p)
            pid = pac_id(p)
            najdeno = "непознато"
            if h is not None and pid:
                for ime, mapa in verzii.items():
                    if pid in mapa:
                        _, _, h2 = proveri_nifti(mapa[pid])
                        if h2 is not None and oblik(h) == oblik(h2) and all(
                            abs(a - b) < 1e-3 for a, b in zip(h["pixdim"][1:4], h2["pixdim"][1:4])
                        ):
                            najdeno = ime
                            break
            sovpaganje[najdeno] += 1
        for ime, c in sovpaganje.most_common():
            info(f"  {c:3d} одговараат на: {ime}")
        if len(sovpaganje) > 1 or "непознато" in sovpaganje:
            vnimanie("patientNN_T1 датотеките не се сите од иста верзија (или се обработени понатаму)")
        info("RadFM треба да ја користи истата верзија како другите модели.")
    else:
        info("Не најдов patientNN_T1.nii* датотеки (форматот од трудот). Проверете која T1 верзија")
        info("(raw / coregistered / MNI) ја користеа MedGemma и M3D, и користете ја истата.")

    # MS ознаки
    naslov("MS ознаки (тип на MS)")
    info_csv = koren / "cross_sectional" / "patient_info.csv"
    vistinski = {}
    if info_csv.is_file():
        with open(info_csv, newline="", encoding="utf-8") as f:
            for red in csv.DictReader(f):
                try:
                    vistinski[f"{int(red['patient_id']):02d}"] = red["ms_type"].strip()
                except (KeyError, ValueError):
                    pass
        dist = collections.Counter(vistinski.values())
        info("open_ms_data/cross_sectional/patient_info.csv: "
             + ", ".join(f"{k} {v}" for k, v in dist.most_common()))
        bez_t1 = sorted(set(vistinski) - set(raw))
        if bez_t1:
            greska("пациенти во patient_info.csv без T1 во raw/: " + ", ".join(f"patient{k}" for k in bez_t1))
    else:
        vnimanie(f"нема {info_csv} — не можам да ги проверам MS ознаките")

    if ms_oznaki:
        sporedi_ms_oznaki(Path(ms_oznaki).expanduser(), vistinski)
    return raw


def normaliziraj_tip(v):
    s = re.sub(r"[\s\-_]", "", str(v)).upper()
    s = re.sub(r"^(SM|MS)", "", s)
    s = re.sub(r"(SM|MS)$", "", s)
    return {"RELAPSINGREMITTING": "RR", "SECONDARYPROGRESSIVE": "SP",
            "PRIMARYPROGRESSIVE": "PP"}.get(s, s or "N/A")


def sporedi_ms_oznaki(pateka, vistinski):
    info(f"Споредба со вашите ознаки: {pateka}")
    if not pateka.is_file():
        greska(f"нема датотека {pateka}")
        return
    with open(pateka, newline="", encoding="utf-8-sig") as f:
        primerok = f.read(4096)
        f.seek(0)
        try:
            dialekt = csv.Sniffer().sniff(primerok, delimiters=",;\t")
        except csv.Error:
            dialekt = csv.excel
        redovi = list(csv.DictReader(f, dialect=dialekt))
    if not redovi:
        greska("датотеката е празна")
        return
    koloni = list(redovi[0].keys())
    id_k = next((k for k in koloni if k and re.search(r"patient|subject|^id$|_id$", k, re.I)), None)
    tip_k = next((k for k in koloni if k and re.search(r"type|tip|phenotype|course", k, re.I)), None)
    info(f"Колони: {koloni}")
    if not id_k or not tip_k or id_k == tip_k:
        vnimanie("не најдов колона за пациент и/или за тип на MS — споредбата е прескокната")
        return
    info(f"Користам: пациент = '{id_k}', тип = '{tip_k}'")
    vasi = collections.defaultdict(set)
    for r in redovi:
        m = re.search(r"(\d+)", str(r.get(id_k, "")))
        if m:
            vasi[f"{int(m.group(1)):02d}"].add(normaliziraj_tip(r.get(tip_k, "")))
    info("Ваша дистрибуција (по пациент): " + ", ".join(
        f"{k} {v}" for k, v in collections.Counter(
            "/".join(sorted(s)) for s in vasi.values()).most_common()))
    zaednicki = sorted(set(vasi) & set(vistinski))
    if len(set(vasi) - set(vistinski)) > 0:
        vnimanie(f"{len(set(vasi) - set(vistinski))} пациенти во вашата датотека ги нема во open_ms_data "
                 f"(на пр. ID {sorted(set(vasi) - set(vistinski))[:5]}) — можеби ознаките се од друга збирка")
    razliki = [(k, vistinski[k], "/".join(sorted(vasi[k]))) for k in zaednicki
               if {vistinski[k]} != vasi[k]]
    if not razliki and zaednicki:
        ok(f"Сите {len(zaednicki)} заеднички пациенти имаат ист тип како во open_ms_data")
    elif razliki:
        vnimanie(f"{len(razliki)} / {len(zaednicki)} пациенти имаат различен тип од open_ms_data:")
        for k, v, w in razliki[:30]:
            info(f"- patient{k}: open_ms_data={v}, ваше={w}")


# ---------------------------------------------------------------------
# 4. Споредба со излезите на другите модели
# ---------------------------------------------------------------------
def sporedi_so_rezultati(pateka, ixi, molab, ms):
    naslov("Споредба со постоечките излези на моделите")
    pateka = Path(pateka).expanduser()
    fajlovi = [pateka] if pateka.is_file() else [
        p for p in pateka.rglob("*") if p.is_file() and p.suffix.lower() in (".json", ".jsonl", ".csv", ".tsv", ".txt")]
    if not fajlovi:
        greska(f"не најдов .json/.jsonl/.csv/.txt во {pateka}")
        return
    tekst = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in fajlovi)
    info(f"Прочитани {len(fajlovi)} датотеки со резултати")

    def sporedi(ime, podatoci, najdeni_vo_rez):
        if not podatoci:
            return
        if not najdeni_vo_rez:
            info(f"{ime}: во резултатите нема имиња од ова множество (друга папка?)")
            return
        nedostasuva = sorted(set(podatoci) - najdeni_vo_rez)
        visok = sorted(najdeni_vo_rez - set(podatoci))
        if not nedostasuva and not visok:
            ok(f"{ime}: истите {len(podatoci)} датотеки како во резултатите")
            return
        if nedostasuva:
            vnimanie(f"{ime}: {len(nedostasuva)} ги имате, а ги нема во резултатите, на пр. {nedostasuva[:3]}")
        if visok:
            greska(f"{ime}: {len(visok)} се во резултатите, а ги немате, на пр. {visok[:3]}")

    sporedi("IXI", ixi, {s for s in ixi if s in tekst}
            | set(re.findall(r"IXI\d{3}-[A-Za-z]+-\d{4}-T1", tekst)))
    sporedi("MOLAB", molab, {s for s in molab if s in tekst}
            | set(re.findall(r"\d{6}__[^\"'\s,;/\\]*?T1Gd_img", tekst)))
    sporedi("MS", {f"patient{k}" for k in ms},
            {f"patient{m}" for m in re.findall(r"patient(\d{2})(?!\d)", tekst)})


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Проверка на МАНУ податоците")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"),
                    help="папка со ixi/, molab/, open_ms_data/ (стандардно ~/manu/data)")
    ap.add_argument("--full", action="store_true",
                    help="целосно откомпресирај секој .gz (посигурно, трае неколку минути)")
    ap.add_argument("--rezultati", help="датотека или папка со излезите на MedGemma/M3D")
    ap.add_argument("--ms-oznaki", help="вашиот CSV со MS ознаки (на пр. clinical_data.csv)")
    a = ap.parse_args()

    data = Path(a.data).expanduser()
    print(f"Проверка на {data}" + (" (целосна)" if a.full else " (брза)"))
    if not data.is_dir():
        print(f"Нема папка {data}. Наведете ја со --data.")
        sys.exit(1)

    ixi = proveri_ixi(data, a.full)
    molab = proveri_molab(data, a.full)
    ms = proveri_ms(data, a.full, a.ms_oznaki)
    if a.rezultati:
        sporedi_so_rezultati(a.rezultati, ixi, molab, ms)

    naslov("РЕЗИМЕ")
    print(f"  Грешки: {len(GRESKI)}   Предупредувања: {len(VNIMANIE)}")
    for g in GRESKI:
        print(f"  - [ГРЕШКА] {g.rstrip(':')}")
    for v in VNIMANIE:
        print(f"  - [ВНИМАНИЕ] {v.rstrip(':')}")
    if not GRESKI:
        print("  Податоците се комплетни според изворот (архиви, figshare md5, git).")
    sys.exit(1 if GRESKI else 0)


if __name__ == "__main__":
    main()

