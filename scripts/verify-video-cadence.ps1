[CmdletBinding()]
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string] $Path,

  [double] $ExpectedFrameRate = 60
)

$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture

function Convert-RationalToDouble([string] $Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) { return [double]::NaN }
  $parts = $Value.Split('/')
  if ($parts.Count -eq 2) {
    $denominator = [double]::Parse($parts[1], $culture)
    if ($denominator -eq 0) { return [double]::NaN }
    return [double]::Parse($parts[0], $culture) / $denominator
  }
  return [double]::Parse($Value, $culture)
}

function Convert-ToDouble($Value) {
  if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string] $Value)) {
    return [double]::NaN
  }
  return [double]::Parse([string] $Value, $culture)
}

function Get-Mean([double[]] $Values) {
  if ($Values.Count -eq 0) { return 0.0 }
  return ($Values | Measure-Object -Average).Average
}

function Get-Percentile([double[]] $Values, [double] $Percentile) {
  if ($Values.Count -eq 0) { return 0.0 }
  $sorted = @($Values | Sort-Object)
  if ($sorted.Count -eq 1) { return [double] $sorted[0] }
  $position = ($sorted.Count - 1) * $Percentile
  $lower = [int] [Math]::Floor($position)
  $upper = [int] [Math]::Ceiling($position)
  if ($lower -eq $upper) { return [double] $sorted[$lower] }
  $weight = $position - $lower
  return [double] $sorted[$lower] * (1 - $weight) + [double] $sorted[$upper] * $weight
}

function Get-MaximumRun([bool[]] $Matches) {
  $maximum = 0
  $current = 0
  foreach ($match in $Matches) {
    if ($match) {
      $current += 1
      if ($current -gt $maximum) { $maximum = $current }
    } else {
      $current = 0
    }
  }
  return $maximum
}

if ($ExpectedFrameRate -le 0 -or [double]::IsNaN($ExpectedFrameRate)) {
  throw 'ExpectedFrameRate must be positive.'
}

$resolvedPath = (Resolve-Path -LiteralPath $Path).Path
foreach ($command in @('ffprobe', 'ffmpeg')) {
  if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
    throw "$command is required and was not found on PATH."
  }
}

$probeText = & ffprobe `
  -v error `
  -select_streams v:0 `
  -count_frames `
  -show_streams `
  -show_frames `
  -show_entries 'stream=codec_name,profile,codec_tag_string,width,height,r_frame_rate,avg_frame_rate,time_base,duration,nb_frames,nb_read_frames,bit_rate:frame=best_effort_timestamp_time,pkt_duration_time,duration_time,key_frame,pict_type' `
  -of json `
  $resolvedPath | Out-String
if ($LASTEXITCODE -ne 0) { throw 'ffprobe failed.' }

$probe = $probeText | ConvertFrom-Json
if (-not $probe.streams -or $probe.streams.Count -eq 0) {
  throw 'No video stream was found.'
}
$stream = $probe.streams[0]
$frames = @($probe.frames)
$timestamps = @(
  $frames |
    ForEach-Object { Convert-ToDouble $_.best_effort_timestamp_time } |
    Where-Object { -not [double]::IsNaN($_) }
)
if ($timestamps.Count -lt 2) { throw 'Fewer than two timestamped video frames were found.' }

$expectedInterval = 1.0 / $ExpectedFrameRate
$firstTimestamp = $timestamps[0]
$timestampErrors = for ($index = 0; $index -lt $timestamps.Count; $index += 1) {
  [Math]::Abs($timestamps[$index] - ($firstTimestamp + $index * $expectedInterval))
}
$timestampDeltas = for ($index = 1; $index -lt $timestamps.Count; $index += 1) {
  $timestamps[$index] - $timestamps[$index - 1]
}
$badTimestampIntervals = @($timestampDeltas | Where-Object {
  [Math]::Abs($_ - $expectedInterval) -gt ($expectedInterval * 0.05)
}).Count
$duplicateTimestamps = @($timestampDeltas | Where-Object { $_ -le 0 }).Count

$keyFrameTimestamps = @(
  for ($index = 0; $index -lt $frames.Count; $index += 1) {
    if ([int] $frames[$index].key_frame -eq 1) {
      $value = Convert-ToDouble $frames[$index].best_effort_timestamp_time
      if (-not [double]::IsNaN($value)) { $value }
    }
  }
)
$keyFrameGaps = for ($index = 1; $index -lt $keyFrameTimestamps.Count; $index += 1) {
  $keyFrameTimestamps[$index] - $keyFrameTimestamps[$index - 1]
}

# YDIF is the mean absolute luma difference from the previous decoded frame.
# Heavy spatial downsampling and a light blur suppress pixel-level encoder noise,
# making repeated 30 Hz simulation states in a nominal 60 fps file much easier to see.
$priorErrorPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$ffmpegLines = @(& ffmpeg `
  -hide_banner `
  -nostats `
  -v info `
  -i $resolvedPath `
  -map '0:v:0' `
  -an `
  -vf 'scale=128:72:flags=area,gblur=sigma=1.5,signalstats,metadata=mode=print' `
  -f null `
  NUL 2>&1 | ForEach-Object { [string] $_ })
$ffmpegExitCode = $LASTEXITCODE
$ErrorActionPreference = $priorErrorPreference
if ($ffmpegExitCode -ne 0) { throw 'ffmpeg visual-cadence decode failed.' }

$ydif = @(
  foreach ($line in $ffmpegLines) {
    if ($line -match 'lavfi\.signalstats\.YDIF=([0-9eE+.-]+)') {
      [double]::Parse($Matches[1], $culture)
    }
  }
)
if ($ydif.Count -lt 2) { throw 'ffmpeg did not emit per-frame YDIF measurements.' }

# Frame zero has no predecessor, so exclude it from transition statistics.
$transitions = [double[]] @($ydif | Select-Object -Skip 1)
$transitionMedian = Get-Percentile $transitions 0.5
$nearDuplicateThreshold = [Math]::Max(0.01, $transitionMedian * 0.10)
$nearDuplicateFlags = [bool[]] @($transitions | ForEach-Object { $_ -le $nearDuplicateThreshold })
$nearDuplicateCount = @($nearDuplicateFlags | Where-Object { $_ }).Count
$discontinuityThreshold = [Math]::Max(8.0, $transitionMedian * 20.0)
$discontinuityCount = @($transitions | Where-Object { $_ -gt $discontinuityThreshold }).Count

$evenTransitions = [System.Collections.Generic.List[double]]::new()
$oddTransitions = [System.Collections.Generic.List[double]]::new()
for ($index = 0; $index -lt $transitions.Count; $index += 1) {
  if (($index % 2) -eq 0) { $evenTransitions.Add($transitions[$index]) }
  else { $oddTransitions.Add($transitions[$index]) }
}
$evenMean = Get-Mean $evenTransitions.ToArray()
$oddMean = Get-Mean $oddTransitions.ToArray()
$transitionMean = Get-Mean $transitions
$alternatingContrast = if ($transitionMean -gt 0) {
  [Math]::Abs($evenMean - $oddMean) / $transitionMean
} else { 0.0 }

$declaredAverageFrameRate = Convert-RationalToDouble $stream.avg_frame_rate
$declaredNominalFrameRate = Convert-RationalToDouble $stream.r_frame_rate
$timestampCadencePass = $duplicateTimestamps -eq 0 -and
  $badTimestampIntervals -eq 0 -and
  (($timestampErrors | Measure-Object -Maximum).Maximum -le ($expectedInterval * 0.05))
$visualCadenceWarning = ($nearDuplicateCount / [Math]::Max(1, $transitions.Count)) -gt 0.20 -or
  $alternatingContrast -gt 0.35 -or
  $discontinuityCount -gt 0

$result = [ordered] @{
  file = $resolvedPath
  expectedFrameRate = $ExpectedFrameRate
  encodedTimeline = [ordered] @{
    codec = $stream.codec_name
    profile = $stream.profile
    codecTag = $stream.codec_tag_string
    width = [int] $stream.width
    height = [int] $stream.height
    declaredAverageFrameRate = $declaredAverageFrameRate
    declaredNominalFrameRate = $declaredNominalFrameRate
    timeBase = $stream.time_base
    durationSeconds = Convert-ToDouble $stream.duration
    frameCount = $frames.Count
    readFrameCount = if ($stream.nb_read_frames) { [int] $stream.nb_read_frames } else { $null }
    bitrate = if ($stream.bit_rate) { [long] $stream.bit_rate } else { $null }
    timestampCadencePass = $timestampCadencePass
    duplicateOrRegressingTimestampCount = $duplicateTimestamps
    intervalErrorOverFivePercentCount = $badTimestampIntervals
    maximumTimestampGridErrorMilliseconds = 1000 * ($timestampErrors | Measure-Object -Maximum).Maximum
    medianTimestampIntervalMilliseconds = 1000 * (Get-Percentile ([double[]] $timestampDeltas) 0.5)
    p95TimestampIntervalMilliseconds = 1000 * (Get-Percentile ([double[]] $timestampDeltas) 0.95)
    keyFrameCount = $keyFrameTimestamps.Count
    maximumKeyFrameGapSeconds = if ($keyFrameGaps.Count -gt 0) {
      ($keyFrameGaps | Measure-Object -Maximum).Maximum
    } else { 0.0 }
  }
  visualCadence = [ordered] @{
    decodedFrameCount = $ydif.Count
    transitionCount = $transitions.Count
    lowPassLumaDifferenceMean = $transitionMean
    lowPassLumaDifferenceP05 = Get-Percentile $transitions 0.05
    lowPassLumaDifferenceMedian = $transitionMedian
    lowPassLumaDifferenceP95 = Get-Percentile $transitions 0.95
    nearDuplicateThreshold = $nearDuplicateThreshold
    nearDuplicateTransitionCount = $nearDuplicateCount
    nearDuplicateTransitionRatio = $nearDuplicateCount / [Math]::Max(1, $transitions.Count)
    longestNearDuplicateTransitionRun = Get-MaximumRun $nearDuplicateFlags
    discontinuityThreshold = $discontinuityThreshold
    discontinuousTransitionCount = $discontinuityCount
    alternatingTransitionMeanA = $evenMean
    alternatingTransitionMeanB = $oddMean
    alternatingContrast = $alternatingContrast
    warning = $visualCadenceWarning
    note = 'warning=true means low-pass differences show repeated/alternating cadence or an abrupt full-frame discontinuity; inspect the source before calling it genuine continuous 60 Hz motion.'
  }
}

$result | ConvertTo-Json -Depth 6
