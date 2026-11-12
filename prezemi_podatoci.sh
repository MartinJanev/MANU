#!/usr/bin/env bash
# =====================================================================
# МАНУ: преземање на податочните множества директно на GPU кластерот
#
# Се стартува на најавниот јазол (10.10.65.6), со вклучен EduVPN:
#   nohup bash prezemi_podatoci.sh > prezemi_podatoci.log 2>&1 &
#   tail -f prezemi_podatoci.log      # следење (Ctrl+C го прекинува само следењето)
#
# Со nohup симнувањето продолжува и ако падне VPN/SSH конекцијата.
# Скриптата може да се стартува повторно: она што веќе е симнато се прескокнува,
# а на крајот секогаш печати преглед (корисно и откако ќе се качи MOLAB).
#
# Структура (по упатството на osc-lab: code/ data/ results/ во една папка):
#   ~/manu/data/ixi/T1/          IXI T1 волумени (цела IXI-T1 архива)
#   ~/manu/data/open_ms_data/    git клон на muschellij2/open_ms_data
#   ~/manu/data/molab/nifti/     MOLAB: рачно качена (ја користи run_radfm.py)
#   ~/manu/data/molab/figshare/  MOLAB: NIfTI слики + клинички податоци од figshare (~8 GB)
#   ~/manu/code  ~/manu/results  ~/manu/containers
# =====================================================================



set -euo pipefail

# BASE="${BASE:-$HOME/manu}"
BASE="${BASE:-$(cd "$(dirname "$0")" && pwd)}" # ова ни помага ако се мења manu фолдерот да се адаптира автоматски
DATA="$BASE/data"
mkdir -p "$DATA/ixi/T1" "$DATA/molab" "$BASE/code" "$BASE/results" "$BASE/containers"

echo "== $(date '+%F %T')  Почеток, BASE=$BASE"

# ---------------------------------------------------------------------
# 0. Дали најавниот јазол има излез кон интернет?
# ---------------------------------------------------------------------
if ! curl -sSfI --max-time 20 https://github.com >/dev/null 2>&1; then
  echo "!! Кластерот нема излез кон интернет (или е блокиран)."
  echo "!! Симнете ги податоците локално и префрлете ги со WinSCP / scp / pscp."
  exit 1
fi
echo "== Интернет пристап: OK"

echo "== Слободен простор:"
df -h "$BASE" || true

# Листа на содржината на архива (патека<TAB>големина), се чува по бришењето на архивата
# за a.py да провери дали е сè распакувано. __MACOSX/ (смет од macOS) се прескокнува.
lista() {
  python3 - "$1" <<'EOF'
import sys, tarfile, zipfile
a = sys.argv[1]
if zipfile.is_zipfile(a):
    clenovi = [(i.filename, i.file_size) for i in zipfile.ZipFile(a).infolist() if not i.is_dir()]
else:
    clenovi = [(m.name, m.size) for m in tarfile.open(a) if m.isfile()]
for ime, gol in clenovi:
    if not ime.startswith("__MACOSX/"):
        print(f"{ime}\t{gol}")
EOF
}

# ---------------------------------------------------------------------
# 1. IXI: целата IXI-T1 архива (здрави контроли)
#    md5 за IXI-T1.tar е земен од torchio (torchio.datasets.ixi)
# ---------------------------------------------------------------------
IXI_URL="http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
IXI_MD5="34901a0593b41dd19c1a1f746eac2d58"
cd "$DATA/ixi"

n_ixi=$(find T1 -name "*.nii.gz" | wc -l)
if [ "$n_ixi" -ge 581 ]; then
  echo "== IXI: T1/ веќе има $n_ixi волумени, прескокнувам симнување."
elif [ ! -s IXI-T1.list ]; then
  echo "== IXI: симнување IXI-T1.tar (неколку GB, може да трае)..."
  wget -c --progress=dot:giga "$IXI_URL/IXI-T1.tar"
  # лоша датотека се брише, инаку wget -c следниот пат би ја „продолжил“ истата грешка
  echo "$IXI_MD5  IXI-T1.tar" | md5sum -c - || { rm -f IXI-T1.tar; echo "!! IXI: md5 не одговара, избришано, пуштете пак."; exit 1; }
  lista IXI-T1.tar > IXI-T1.list.tmp
  tar -xf IXI-T1.tar -C T1
  mv IXI-T1.list.tmp IXI-T1.list
  rm -f IXI-T1.tar
else
  echo "== IXI: IXI-T1.tar е веќе симнат и распакуван ($(wc -l < IXI-T1.list) датотеки), прескокнувам."
fi

# Демографија (опционално, не е критично ако не успее)
[ -f IXI.xls ] || wget -q "$IXI_URL/IXI.xls" || echo "-- IXI.xls не се симна (не е критично)."

# ---------------------------------------------------------------------
# 2. open_ms_data (MS, Ljubljana LIT, CC-BY): цело репо, последна верзија
# ---------------------------------------------------------------------
cd "$DATA"
if [ -d open_ms_data/.git ]; then
  echo "== open_ms_data: git pull..."
  git -C open_ms_data pull --ff-only
  if git lfs version >/dev/null 2>&1; then git -C open_ms_data lfs pull; fi
elif [ -d open_ms_data ]; then
  echo "== open_ms_data: папката веќе постои (не е git клон), прескокнувам."
else
  echo "== open_ms_data: git clone..."
  git clone --depth 1 https://github.com/muschellij2/open_ms_data.git
  if git lfs version >/dev/null 2>&1; then git -C open_ms_data lfs pull; fi
fi

# Ако .nii.gz фајловите се по неколку стотини бајти, тоа се git-lfs покажувачи
first_nii=$(find open_ms_data/cross_sectional -name '*.nii.gz' 2>/dev/null | head -n 1 || true)
if [ -n "$first_nii" ] && [ "$(stat -c %s "$first_nii")" -lt 10000 ]; then
  echo "!! $first_nii е мал ($(stat -c %s "$first_nii") B): најверојатно git-lfs покажувач."
  echo "!! Пробајте:  cd $DATA/open_ms_data && git lfs pull"
fi

# ---------------------------------------------------------------------
# 3. MOLAB / OpenBTAI (Ocaña-Tienda et al. 2023): цела figshare колекција
#    https://doi.org/10.6084/m9.figshare.c.6194104  (~46 GB: DICOM, NIfTI, маски, xlsx)
#    Листата на датотеки (име, големина, md5) се зема од figshare API, не од трудот.
#    Постоечката molab/nifti/ (што ја користи run_radfm.py) не се менува;
#    a.py ја споредува со molab/figshare/NIFTIs_Images/.
# ---------------------------------------------------------------------
FS="$DATA/molab/figshare"
mkdir -p "$FS"
cd "$FS"
# Само двата figshare артикли што ни требаат (~8 GB наместо ~46 GB):
#   22272109 = "Images in nifti format" (NIFTIs_Images.zip: точно 373 слики, 75 пациенти)
#   22270933 = "Clinical data" (OpenBTAI_METS_ClinicalData.xlsx: примарно потекло)
# DICOM сетовите, маските и радиомиката не ги користиме.
for id in 22272109 22270933; do
  curl -sSf "https://api.figshare.com/v2/articles/$id" | python3 -c '
import json, sys
for f in json.load(sys.stdin)["files"]:
    print(f["name"], f["size"], f["computed_md5"], f["download_url"], sep="\t")'
done > manifest.tsv.tmp
mv manifest.tsv.tmp manifest.tsv
echo "== MOLAB: figshare има $(wc -l < manifest.tsv) датотеки"

while IFS=$'\t' read -r ime golemina md5 url; do
  if [ -s "$ime.list" ] || { [ -f "$ime" ] && [ "${ime%.zip}" = "$ime" ] && [ "$(stat -c %s "$ime")" = "$golemina" ]; }; then
    echo "== MOLAB: $ime веќе е симнат, прескокнувам."
    continue
  fi
  echo "== MOLAB: симнување $ime ($((golemina / 1024 / 1024)) MB)..."
  wget -c --progress=dot:giga -O "$ime" "$url"
  echo "$md5  $ime" | md5sum -c - || { rm -f "$ime"; echo "!! MOLAB: $ime md5 не одговара, избришано, пуштете пак."; exit 1; }
  if [ "${ime%.zip}" != "$ime" ]; then
    lista "$ime" > "$ime.list.tmp"
    python3 -m zipfile -e "$ime" .
    rm -rf __MACOSX
    mv "$ime.list.tmp" "$ime.list"
    rm -f "$ime"
  fi
done < manifest.tsv

# ---------------------------------------------------------------------
# 4. Преглед: споредба со изворот (листи на архивите, md5, git)
# ---------------------------------------------------------------------
echo
du -sh "$DATA"/* 2>/dev/null || true
echo "== $(date '+%F %T')  Готово. Проверка:"
python3 "$BASE/proveri_podatoci.py" --data "$DATA" || true
