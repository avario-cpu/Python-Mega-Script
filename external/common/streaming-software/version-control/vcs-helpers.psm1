try {
  . "$PSScriptRoot\dotsource-common-paths.ps1"
} catch {
  Write-Host "Failed to load common paths module: $($_.Exception.Message)" -ForegroundColor Red
  throw
}

$script:VcsLogDirPath = $null
$script:VcsMaxLogSizeBytes = 2MB
$script:VcsVerboseEnabled = $false
$Script:LogSuffix = "-vcs-templater.log"

function Get-VcsMarkerFile {
  param([Parameter(Mandatory=$true)] [string]$DirectoryPath)
  return Get-ChildItem $DirectoryPath -File -Filter "*-marker.json" | Select-Object -First 1
}

function Remove-EmptyVcsDirectories {
  param(
    [Parameter(Mandatory=$true)][string]$RootPath,
    [Parameter(Mandatory=$false)][string]$RequiredAncestorPattern
  )
  if (-not (Test-Path $RootPath -PathType Container)) {
    Write-Warning "Remove-EmptyVcsDirectories: RootPath does not exist or is not a directory: $RootPath"
    return
  }

  $subDirs = Get-ChildItem $RootPath -Directory -Recurse |
    Sort-Object { $_.FullName.Length } -Descending
  if ($RequiredAncestorPattern) {
    $matchedDirs = $subDirs | Where-Object { $_.FullName -match $RequiredAncestorPattern }
    if ($subDirs.Count -gt 0 -and $matchedDirs.Count -eq 0) {
      Write-Warning ("Remove-EmptyVcsDirectories: RequiredAncestorPattern matched no " `
          + "directories under: $RootPath")
    }
    $subDirs = $matchedDirs
  }
  foreach ($dir in $subDirs) {
    if (-not (Get-ChildItem $dir.FullName -Force)) {
      Remove-Item $dir.FullName -Force
      Write-VcsMessage -Message "  Removed empty vcs directory: $($dir.FullName)" -Color Yellow
    }
  }
}

function Test-DanglingVcsSymlink {
  param([Parameter(Mandatory=$true)] [System.IO.FileInfo]$SymlinkFile)
  if (-not $SymlinkFile.LinkType) {
    return $false
  }
  $target = $SymlinkFile.LinkTarget
  if (-not $target) {
    return $true
  }
  return -not (Test-Path $target)
}

function Write-VcsLogSeparator {
  param([int]$Lines = 5)
  if ($script:VcsLogFilePath) {
    try {
      $blankLines = 1..$Lines | ForEach-Object { "" }
      $blankLines + "new_run_started" |
        Add-Content -Path $script:VcsLogFilePath -Encoding UTF8
    } catch {
      Write-Warning "VCS logging failed: $_"
    }
  }
}

function Set-VcsVerbose {
  param([switch]$Enabled)
  $script:VcsVerboseEnabled = [bool]$Enabled
}

function Set-VcsLogFilePath {
  param(
    [Parameter(Mandatory=$true)][string]$LogDirPath,
    [Parameter(Mandatory=$true)][string]$AppName
  )
  if ([string]::IsNullOrWhiteSpace($logDirPath)) {
    throw "logDirPath cannot be null or empty."
  }
  try {
    [System.IO.Path]::GetFullPath($LogDirPath)
  } catch {
    throw "LogDirPath is not a valid path: '$LogDirPath'"
  }

  $logPath = Join-Path $LogDirPath "$AppName$LogSuffix"

  $resolvedDir = Split-Path $logPath -Parent
  if (-not (Test-Path $resolvedDir)) {
    New-Item -ItemType Directory -Path $resolvedDir -Force | Out-Null
  }

  if ((Test-Path $logPath) -and
    (Get-Item $logPath).Length -gt $script:VcsMaxLogSizeBytes) {
    $lines = Get-Content $logPath
    $keepFrom = [Math]::Floor($lines.Count / 2)
    $lines | Select-Object -Skip $keepFrom | Set-Content $logPath -Encoding UTF8
  }

  $script:VcsLogDirPath  = $resolvedDir
  $script:VcsLogFilePath = $logPath
}

function Write-VcsMessage {
  param(
    [Parameter(Mandatory=$true)] [AllowEmptyString()] [string]$Message,
    [string]$Color,
    [switch]$AsVerbose,
    [switch]$NoLog
  )
  if ($AsVerbose) {
    if ($script:VcsVerboseEnabled) {
      Write-Host $Message -ForegroundColor DarkGray
    }
  } elseif ($Color) {
    Write-Host $Message -ForegroundColor $Color
  } else {
    Write-Host $Message
  }
  if ($script:VcsLogFilePath -and -not $NoLog) {
    try {
      $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
      "[$timestamp] $Message" |
        Add-Content -Path $script:VcsLogFilePath -Encoding UTF8
    } catch {
      Write-Warning "VCS logging failed: $_"
    }
  }
}

function Assert-HelpersPaths {
  $helperPaths = @(
    $Global:RepoPath
    $script:PrettierPath
  )
  foreach ($path in $helperPaths) {
    if (-not (Test-Path $path)) {
      Write-Warning "Required path not found: '$(Split-Path $path -Leaf)'" `

      if ($path -eq $RepoPath) {
        Write-Warning "Global:RepoPath is not set correctly"
      }

      if ($path -eq $PrettierPath) {
        Write-Warning "Prettier is not installed or not found at the expected path: $path"
        Write-Warning "check $CommonVcsPaths"
      }

      Write-Warning "check $CommonVcsPaths"
      Write-ThrowContext
      throw "Required path not found: $path"
    }
  }
}

function ConvertFrom-Json5 {
  param([string]$Path)

  $json5Path = (Join-Path $RepoPath "node_modules\json5") -replace '\\', '/'

  # Escape backslashes so the path survives as a JS string literal
  $safePath = $Path -replace '\\', '/'

  $json = node -e "
    const JSON5 = require('$json5Path');
    const fs = require('fs');
    console.log(JSON.stringify(
      JSON5.parse(fs.readFileSync('$safePath','utf8'))
    ));
  "

  return $json | ConvertFrom-Json -AsHashtable
}

function Read-MappingsFile {
  param([Parameter(Mandatory=$true)][string]$Path)

  if (-not (Test-Path $Path)) {
    Write-ThrowContext
    throw "Mappings file not found: $Path"
  }

  $raw = ConvertFrom-Json5 $Path
  $rules = @()

  foreach ($category in $raw.Keys) {
    foreach ($token in $raw[$category].Keys) {
      $rules += [PSCustomObject]@{
        Key   = $null
        Value = [string]$raw[$category][$token]
        Token = $token
      }
    }
  }

  return $rules
}

function Read-ScopedMappingsFile {
  param([Parameter(Mandatory=$true)][string]$Path)

  if (-not (Test-Path $Path)) {
    Write-ThrowContext
    throw "Scoped mappings file not found: $Path"
  }

  $raw = ConvertFrom-Json5 $Path

  if (-not $raw.ContainsKey("scoped")) {
    Write-ThrowContext
    throw "Scoped mappings file is missing a top-level 'scoped' array: $Path"
  }
  if (-not $raw.scoped -or $raw.scoped.Count -eq 0) {
    Write-ThrowContext
    throw "Scoped mappings file has an empty 'scoped' array: $Path"
  }

  $rules = @()
  foreach ($entry in $raw.scoped) {
    if (-not $entry.key -or -not $entry.value -or -not $entry.token) {
      Write-ThrowContext
      throw "Scoped mapping entry missing key/value/token: $($entry | ConvertTo-Json -Compress)"
    }
    $rules += [PSCustomObject]@{
      Key   = [string]$entry.key
      Value = [string]$entry.value
      Token = [string]$entry.token
    }
  }

  return $rules
}

function Read-ReplacementMappings {
  param(
    [string]$CommonMappingsPath,
    [string]$MappingsPath,
    [Parameter(Mandatory=$true)] [string[]]$ScopedMappingsPaths
  )

  if (-not $ScopedMappingsPaths -or $ScopedMappingsPaths.Count -eq 0) {
    Write-ThrowContext
    throw "Read-ReplacementMappings requires at least one -ScopedMappingsPaths entry `
        (e.g. a ports file)."
  }

  $rules = @()

  if ($CommonMappingsPath) {
    if (Test-Path $CommonMappingsPath) {
      $rules += Read-MappingsFile $CommonMappingsPath
    } else {
      Write-Warning "CommonMappingsPath was specified but not found: $CommonMappingsPath"
    }
  }

  if ($MappingsPath) {
    if (Test-Path $MappingsPath) {
      $rules += Read-MappingsFile $MappingsPath
    } else {
      Write-Warning "MappingsPath was specified but not found: $MappingsPath"
    }
  }

  foreach ($path in $ScopedMappingsPaths) {
    $rules += Read-ScopedMappingsFile $path
  }

  if ($rules.Count -eq 0) {
    Write-ThrowContext
    throw "Read-ReplacementMappings produced zero rules — check your mapping file paths and contents."
  }

  # Longest value first — so a full URL is matched before a bare port
  # substring that happens to be embedded inside it.
  return $rules | Sort-Object { $_.Value.Length } -Descending
}

function Format-JsonWithPrettier {
  param([string[]]$FilePaths)

  if (-not $Script:PrettierPath) {
    Write-VcsMessage -Message "Warning: Prettier path is not set. Skipping formatting." -Color Yellow
    return
  }

  if (-not (Test-Path $script:PrettierPath)) {
    Write-VcsMessage -Message "Warning: Prettier not found at $script:PrettierPath. Skipping formatting." `
      -Color Yellow
    return
  }

  Write-VcsMessage -AsVerbose -Message "Formatting $($FilePaths.Count) file(s) with Prettier..."

  $output = & $script:PrettierPath --write $FilePaths --no-config 2>&1
  foreach ($line in $output) {
    if ($line -is [System.Management.Automation.ErrorRecord]) {
      Write-VcsMessage -Message "  $line" -Color Red
    } else {
      Write-VcsMessage -AsVerbose -Message "  $line"
    }
  }
  if ($LASTEXITCODE -ne 0) {
    Write-VcsMessage -Message "Warning: Prettier exited with code $LASTEXITCODE" -Color Red
    $output | ForEach-Object { Write-VcsMessage -Message "  $_" -Color Red }
  }
}

function Assert-InputPath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [Parameter(Mandatory = $true)]
    [array]$Roots
  )

  $normalizedPath = $Path.TrimEnd('\')

  $valid = $Roots | Where-Object {
    $normalizedPath.StartsWith(
      $_.Path.TrimEnd('\'),
      [System.StringComparison]::OrdinalIgnoreCase
    )
  }

  if (-not $valid) {
    Write-VcsMessage -Message "This function must target files under:" -Color Red
    $Roots.Path | ForEach-Object { Write-VcsMessage -Message "  $_" -Color Red }
    Write-VcsMessage -Message "Current target: $Path" -Color Red
    Write-ThrowContext
    throw "Invalid target path: $Path"
  }
}

function Get-VcsRelativePath {
  param(
    [Parameter(Mandatory=$true)]
    [string]$InputFilePath,

    [Parameter(Mandatory=$true)]
    [array]$Roots,

    [Parameter(Mandatory=$true)]
    [array]$Markers,

    [Parameter(Mandatory=$true)]
    [string]$AppName
  )

  $prefix = $null

  foreach ($root in $Roots) {
    if ($InputFilePath.StartsWith(
        $root.Path,
        [System.StringComparison]::OrdinalIgnoreCase
      )) {
      $prefix = $root.Name
      $relative = $InputFilePath.Substring(
        $root.Path.Length
      ).TrimStart('\')
      break
    }
  }

  if ($null -eq $prefix) {
    Write-ThrowContext
    throw "Unexpected $AppName import path: $InputFilePath"
  }

  $parts = $relative -split '\\'
  $dirOnlyParts = $parts[0..($parts.Count - 2)]

  foreach ($marker in $Markers) {
    $index = $dirOnlyParts.IndexOf($marker)

    if ($index -ge 0) {
      $remainingCount = $dirOnlyParts.Count - ($index + 1)

      if ($remainingCount -le 0) {
        return Join-Path $prefix $marker
      }

      $dirParts = $dirOnlyParts[
      ($index + 1)..($dirOnlyParts.Count - 1)
      ]

      return Join-Path $prefix (
        Join-Path $marker ($dirParts -join '\')
      )
    }
  }

  Write-VcsMessage -Message "Expected config locations:" -Color Red
  foreach ($marker in $Markers) {
    Write-VcsMessage -Message "  <root>\$marker\..." -Color Red
  }
  throw (
    "Unexpected $AppName config location: $relative"
  )
}

function Invoke-ScopedReplace {
  param(
    [Parameter(Mandatory=$true)] [string]$Content,
    [Parameter(Mandatory=$true)] [string]$Key,
    [Parameter(Mandatory=$true)] [string]$SearchValue,
    [Parameter(Mandatory=$true)] [string]$Token
  )

  $isNumericSearch = $SearchValue -match '^-?\d+(\.\d+)?$'
  $countRef = [ref]0

  # "Key": "quoted value" OR "Key": bareNumber
  $pattern = "(?<prefix>`"$([regex]::Escape($Key))`"\s*:\s*)" +
  "(?<val>`"(?:[^`"\\]|\\.)*`"|-?\d+(?:\.\d+)?)"

  $newContent = [regex]::Replace($Content, $pattern, {
      param($m)
      $prefix = $m.Groups['prefix'].Value
      $val    = $m.Groups['val'].Value

      if ($val.StartsWith('"')) {
        $inner = $val.Substring(1, $val.Length - 2)   # strip surrounding quotes

        if ($isNumericSearch) {
          # Numeric rule values require an exact match even if the target happens to be quoted —
          # never a substring match on digits. Don't replace 8080 in "19808045" or "8080abc": the
          # port number 8080 is only meaningful when on its own.
          if ($inner -eq $SearchValue) {
            Write-VcsMessage -AsVerbose -Message (
              "  Replaced ($Key): `"$SearchValue`" -> `"$Token`"") -Color DarkCyan
            $countRef.Value++
            return $prefix + "`"$Token`""
          }
        } elseif ($inner.Contains($SearchValue)) {
          Write-VcsMessage -AsVerbose -Message "  Replaced ($Key): $SearchValue -> $Token in $val" `
            -Color DarkCyan
          $countRef.Value++
          return $prefix + "`"$($inner.Replace($SearchValue, $Token))`""
        }
      } elseif ($val -eq $SearchValue) {
        Write-VcsMessage -AsVerbose -Message "  Replaced ($Key): $SearchValue -> `"$Token`"" `
          -Color DarkCyan
        $countRef.Value++
        return $prefix + "`"$Token`""
      }

      return $m.Value
    }.GetNewClosure())

  return [PSCustomObject]@{ Content = $newContent; Count = $countRef.Value }
}

function Invoke-ScopedRestore {
  param(
    [Parameter(Mandatory=$true)] [string]$Content,
    [Parameter(Mandatory=$true)] [string]$Value,
    [Parameter(Mandatory=$true)] [string]$Token
  )

  $isNumeric = $Value -match '^-?\d+(\.\d+)?$'
  $count = 0

  if ($isNumeric) {
    $quotedToken = "`"$Token`""
    if ($Content.Contains($quotedToken)) {
      $Content = $Content.Replace($quotedToken, $Value)
      Write-VcsMessage -AsVerbose -Message "  Replaced: $quotedToken -> $Value" -Color DarkCyan
      $count++
    }
  } elseif ($Content.Contains($Token)) {
    $escapedValue = ($Value | ConvertTo-Json -Compress).Trim('"')
    $Content = $Content.Replace($Token, $escapedValue)
    Write-VcsMessage -AsVerbose -Message "  Replaced: $Token -> $Value" -Color DarkCyan
    $count++
  }

  return [PSCustomObject]@{ Content = $Content; Count = $count }
}

function ConvertTo-VcsTemplateFile {
  param(
    [Parameter(Mandatory=$true)] [string]$InputFilePath,
    [Parameter(Mandatory=$true)] [string]$VcsOutDirPath,
    [Parameter(Mandatory=$true)] [array]$Rules,
    [Parameter(Mandatory=$true)]
    [AllowEmptyCollection()]
    [System.Collections.Generic.List[string]]$FormatQueue,
    [Parameter(Mandatory=$true)] [ref]$ReplacedCount
  )

  $inputFileName  = Split-Path $InputFilePath -Leaf
  $inputDirectory = Split-Path $InputFilePath -Parent

  if ($InputFilePath -notmatch '\.json$') {
    Write-ThrowContext
    throw "Input file must be a .json file, got: $inputFileName"
  }

  $templateFileName = $inputFileName -replace "\.json$", ".vcs-template.json"
  $vcsOutFilePath   = Join-Path $VcsOutDirPath $templateFileName

  Write-VcsMessage -AsVerbose -Message "Creating vcs template from real config..."
  Write-VcsMessage -AsVerbose -Message "Input:  $InputFilePath"
  Write-VcsMessage -AsVerbose -Message "Output: $vcsOutFilePath"

  if (-not (Test-Path $VcsOutDirPath)) {
    New-Item -ItemType Directory -Path $VcsOutDirPath -Force | Out-Null
    Write-VcsMessage -Message "Created VCS directory: $VcsOutDirPath" -Color Yellow
  }

  $symlinkPath = Join-Path $inputDirectory $templateFileName
  if (Test-Path $symlinkPath) {
    Remove-Item $symlinkPath -Force
  }

  $content = Get-Content $InputFilePath -Raw
  $sortedRules = $Rules | Sort-Object { $_.Value.Length } -Descending

  foreach ($rule in $sortedRules) {
    if ($rule.Key) {
      $result = Invoke-ScopedReplace `
        -Content $content `
        -Key $rule.Key `
        -SearchValue $rule.Value `
        -Token $rule.Token
      $content = $result.Content
      $ReplacedCount.Value += $result.Count
    } else {
      $variants = @(
        $rule.Value,
        ($rule.Value | ConvertTo-Json -Compress).Trim('"')
      )
      foreach ($variant in $variants) {
        if ($content.Contains($variant)) {
          $content = $content.Replace($variant, $rule.Token)
          Write-VcsMessage -AsVerbose -Message "  Replaced: $variant -> $($rule.Token)" `
            -Color DarkCyan
          $ReplacedCount.Value++
        }
      }
    }
  }

  $needsWrite = $true
  if (Test-Path $vcsOutFilePath) {
    $existingCanonical = (Get-Content $vcsOutFilePath -Raw |
        ConvertFrom-Json -AsHashtable | ConvertTo-Json -Compress -Depth 100)
    $newCanonical = ($content | ConvertFrom-Json -AsHashtable | ConvertTo-Json -Compress -Depth 100)
    $needsWrite = $existingCanonical -ne $newCanonical
  }

  if ($needsWrite) {
    $content | Set-Content $vcsOutFilePath -Encoding UTF8
    Write-VcsMessage -AsVerbose -Message "Template saved: $vcsOutFilePath"
  } else {
    Write-VcsMessage -AsVerbose -Message "Unchanged content: $vcsOutFilePath"
  }
  $FormatQueue.Add($vcsOutFilePath)

  New-Item `
    -ItemType SymbolicLink `
    -Path $symlinkPath `
    -Target $vcsOutFilePath | Out-Null
}

function ConvertFrom-VcsTemplateFile {
  param(
    [Parameter(Mandatory=$true)]  [string]$InputFilePath,
    [Parameter(Mandatory=$true)]  [array]$Rules,
    [Parameter(Mandatory=$false)] [switch]$Backup,
    [Parameter(Mandatory=$true)]  [ref]$ReplacedCount,
    [Parameter(Mandatory=$true)]
    [AllowEmptyCollection()]
    [System.Collections.Generic.List[string]]$FormatQueue
  )

  $inputFileName = Split-Path $InputFilePath -Leaf
  $inputDirectory = Split-Path $InputFilePath -Parent

  if ($InputFilePath -notmatch '\.vcs-template\.json$') {
    Write-ThrowContext
    throw "Input filename must be like **.vcs-template.json, got: $inputFileName"
  }

  $outFilePath = $InputFilePath -replace '\.vcs-template\.json$', '.json'

  Write-VcsMessage -AsVerbose -Message "Restoring real config from template..."
  Write-VcsMessage -AsVerbose -Message "Input:  $InputFilePath"
  Write-VcsMessage -AsVerbose -Message "Output: $outFilePath"

  if ($Backup -and (Test-Path $outFilePath)) {
    $backupPath = "$outFilePath.bak"
    Copy-Item $outFilePath $backupPath -Force
    Write-VcsMessage -Message "Backup saved: $backupPath" -Color Magenta
  }

  $content = Get-Content $InputFilePath -Raw

  if ([string]::IsNullOrWhiteSpace($content)) {
    $markerFile = Get-VcsMarkerFile -DirectoryPath $inputDirectory
    Write-VcsMessage -Message "Template file: $InputFilePath is empty or unreadable" -Color Red
    if ($markerFile) {
      Write-VcsMessage -Message "  (Marker file present: $($markerFile.Name))" -Color Red
    } else {
      Write-VcsMessage -Message "  (No Marker file found)" -Color Red
    }
    Write-ThrowContext
    throw "Template file is empty or unreadable: $InputFilePath"
  }

  foreach ($rule in $Rules) {
    $result = Invoke-ScopedRestore -Content $content -Value $rule.Value -Token $rule.Token
    $content = $result.Content
    $ReplacedCount.Value += $result.Count
  }

  $unresolvedMatches = [regex]::Matches($content, '\{\{[A-Z0-9_]+\}\}') |
    Select-Object -ExpandProperty Value -Unique
  foreach ($unresolved in $unresolvedMatches) {
    Write-VcsMessage -Message "Warning: No mapping found for token $unresolved — left as-is" `
      -Color Yellow
  }

  $content | Set-Content $outFilePath -Encoding UTF8
  $FormatQueue.Add($outFilePath)
  Write-VcsMessage -AsVerbose -Message "Real config saved: $outFilePath"

  return $outFilePath
}

$FunctionsToExport = @(
  "Assert-InputPath"
  "ConvertFrom-VcsTemplateFile"
  "ConvertTo-VcsTemplateFile"
  "Format-JsonWithPrettier"
  "Get-VcsRelativePath"
  "Get-VcsMarkerFile"
  "Read-ReplacementMappings"
  "Remove-EmptyVcsDirectories"
  "Set-VcsLogFilePath"
  "Set-VcsVerbose"
  "Test-DanglingVcsSymlink"
  "Write-VcsLogSeparator"
  "Write-VcsMessage"
)

Assert-HelpersPaths
Export-ModuleMember -Function $FunctionsToExport


















