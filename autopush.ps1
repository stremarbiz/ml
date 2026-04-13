Set-Location "C:\Users\pytho\Desktop\C\ML"

while ($true) {
    $changes = git status --porcelain

    if ($changes) {
        git add .
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit -m "auto: $timestamp"
        git push origin live
    }

    Start-Sleep -Seconds 3
}