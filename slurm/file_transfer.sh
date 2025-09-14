#!/usr/bin/env bash
# scp_transfer.sh
# Usage:
#   ./scp_transfer.sh -c cipName -s src_path [-d dest]
# If an option is omitted the script will prompt you for it.
set -u

usage() {
  cat <<EOF
Usage: $0 [-c cipName] [-s src] [-d dest]
  -c cipName   CIP name (user@host prefix before @remote.cip.ifi.lmu.de)
  -s src       Source path on remote machine (relative to remote machine)
  -d dest      Destination directory on local machine (default: .)
EOF
  exit 1
}

# parse options
cipName=""
src=""
dest=""

while getopts ":c:s:d:h" opt; do
  case $opt in
    c) cipName="$OPTARG" ;;
    s) src="$OPTARG" ;;
    d) dest="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Prompt for missing values (like Read-Host)
if [[ -z "$cipName" ]]; then
  read -rp "Input your CIP name: " cipName
fi

if [[ -z "$src" ]]; then
  read -rp "Input src path relative to the remote machine: " src
fi

if [[ -z "$dest" ]]; then
  dest="."
fi

if [[ "$dest" == "." ]]; then
  echo "No destination directory specified, using current directory as destination"
fi

fullUrl="${cipName}@remote.cip.ifi.lmu.de:${src}"

echo "Transferring files from $fullUrl into $dest..."
scp -r -- "$fullUrl" "$dest"
scp_exit=$?

if [[ $scp_exit -ne 0 ]]; then
  echo "scp failed with exit code $scp_exit" >&2
  exit $scp_exit
fi

echo "Transfer complete."
