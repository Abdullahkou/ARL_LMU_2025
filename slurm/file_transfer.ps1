param (
    [string]$cipName = $( Read-Host "Input your CIP name" ),
    [string]$src = $( Read-Host "Input src path relative to the remote machine" ),
    [string]$dest = "."
)

if ($dest -eq ".") {
    Write-Host "No destination directory specified, using current directory as destination"
}

$fullUrl = "$cipName@remote.cip.ifi.lmu.de:$src"

Write-Host "Transferring files from $fullUrl into $dest..."

scp -r $fullUrl $dest