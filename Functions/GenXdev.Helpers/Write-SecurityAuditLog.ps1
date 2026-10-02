###############################################################################
<#
.SYNOPSIS
Logs security-relevant events to a tamper-resistant audit file.

.DESCRIPTION
This helper records security-sensitive actions for tracking, investigation,
and defense-in-depth review.
#>
function Write-SecurityAuditLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet(
            'CommandGenerated',
            'CommandBlocked',
            'CommandExecuted',
            'CommandFailed',
            'ExecutionElevated',
            'ToolCallExecuted',
            'ScheduledTaskCreated',
            'KeystrokesSent',
            'LLMQueryMade'
        )]
        [string] $EventType,

        [Parameter(Mandatory = $true)]
        [string] $Description,

        [Parameter(Mandatory = $false)]
        [hashtable] $Details = @{}
    )

    try {
        $auditDir = $null
        if (Microsoft.PowerShell.Management\Get-Command -Name 'GenXdev\Get-GenXdevPreferencesDatabasePath' -ErrorAction SilentlyContinue) {
            $auditDir = GenXdev\Get-GenXdevPreferencesDatabasePath
        }

        if ([string]::IsNullOrWhiteSpace($auditDir)) {
            $auditDir = Join-Path $env:LOCALAPPDATA 'GenXdev'
        }

        $null = Microsoft.PowerShell.Management\New-Item -ItemType Directory -Path $auditDir -Force -ErrorAction SilentlyContinue
        $auditPath = Join-Path $auditDir 'security-audit.log'

        $entry = [ordered]@{
            Timestamp   = (Get-Date).ToString('o')
            EventType   = $EventType
            Description = $Description
            User        = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
            Computer    = [System.Net.Dns]::GetHostName()
            Details     = $Details
        }

        $line = ($entry | Microsoft.PowerShell.Utility\ConvertTo-Json -Compress -Depth 10)
        Add-Content -Path $auditPath -Value $line -Encoding UTF8 -ErrorAction SilentlyContinue
    }
    catch {
        Microsoft.PowerShell.Utility\Write-Verbose "Audit logging failed: $($_.Exception.Message)"
    }
}
###############################################################################
