# PowerShell Script: Create Daily Intelligence Structure in D:/Vault/Vault/

$vaultPath = "D:\Vault\Vault"

# Create main _BRAIN directory structure
$folders = @(
    "_BRAIN",
    "_BRAIN\DAILY_CAPTURES",
    "_BRAIN\DAILY_REPORTS",
    "_BRAIN\WEEKLY_REVIEWS",
    "_BRAIN\MONTHLY_DEEPDIVES",
    "_BRAIN\PATTERNS",
    "_BRAIN\RECOMMENDATIONS",
    "_BRAIN\_TEMPLATES"
)

Write-Host "🚀 Creating Daily Intelligence Structure in $vaultPath`n"

foreach ($folder in $folders) {
    $fullPath = Join-Path $vaultPath $folder
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "✅ Created: $folder"
    } else {
        Write-Host "⏭️  Already exists: $folder"
    }
}

Write-Host "`n📋 Directory structure created successfully!`n"
Write-Host "Your vault is organized at: $vaultPath\_BRAIN"
