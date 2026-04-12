# run_server.ps1 — Safely start the Nexus server, killing any existing process on port 7860
$port = 7860
$pid_line = netstat -ano | Select-String ":$port\s.*LISTENING"
if ($pid_line) {
    $old_pid = ($pid_line -replace '.*\s+(\d+)$', '$1').Trim()
    Write-Host "Killing existing process on port $port (PID $old_pid)..."
    taskkill /PID $old_pid /F | Out-Null
    Start-Sleep -Seconds 1
}
Write-Host "Starting server on port $port..."
python server\app.py
