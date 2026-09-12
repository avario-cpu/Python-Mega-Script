# Ko-fi webhook test sender
# based off unofficial documentation:
# {
#   "verification_token": "your-actual-verification-token",
#   "message_id": "some-uuid",
#   "timestamp": "2026-07-27T16:27:32Z",
#   "type": "Donation", # Donation | Subscription | Commission | Shop Order
#   "is_public": true,
#   "from_name": "Jo Example",
#   "message": "Good luck with the integration!",
#   "amount": "3.00",
#   "url": "https://ko-fi.com/...",
#   "email": "jo@example.com",
#   "currency": "USD",
#   "is_subscription_payment": false,
#   "is_first_subscription_payment": false,
#   "kofi_transaction_id": "some-uuid",
#   "shop_items": null,
#   "tier_name": null, # needs tiers to be configured on Ko-fi for the profile
#   "shipping": null
# }
# Usage:
# Test-KofiWebhook -Type "Donation" -FromName "TestUser" -Message "hi" -Amount "3" -Currency "USD"
# Shorthand presets:
# Test-KofiWebhook sub -Message "message" ...
# Test-KofiWebhook resub -Message "message" ...
# Test-KofiWebhook dono -Message "message" ...

function Set-KofiSecret {
  param(
    [Parameter(Mandatory)]
    [string]$Target
  )

  $secure = Read-Host "Value for $Target" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)

  try {
    cmdkey /generic:$Target /user:kofi /pass:$plain | Out-Null
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $plain = $null
  }

  Write-Host "Stored credential for '$Target'" -ForegroundColor Green
}

function Test-KofiWebhook {
  param(
    [Parameter(Position = 0)]
    [ValidateSet("sub", "resub", "dono")]
    [string]$Preset,

    [ValidateSet("ftp", "prod")]
    [Parameter(Position = 1)]
    [string]$Env = "ftp",

    [string]$Type = "Donation",
    [string]$FromName = "TestUser",
    [string]$Message = "This is a custom test message",
    [string]$Amount = "5.00",
    [string]$Currency = "USD",
    [bool]$IsFirstSubscriptionPayment = $false,
    [bool]$IsPublic = $true
  )

  switch ($Preset) {
    "dono"   {
      if (-not $PSBoundParameters.ContainsKey('Type')) {
        $Type = "Donation"
      }
    }
    "sub"   {
      if (-not $PSBoundParameters.ContainsKey('Type')) {
        $Type = "Subscription"
      }
      if (-not $PSBoundParameters.ContainsKey('IsFirstSubscriptionPayment')) {
        $IsFirstSubscriptionPayment = $true
      }
    }
    "resub" {
      if (-not $PSBoundParameters.ContainsKey('Type')) {
        $Type = "Subscription"
      }
      if (-not $PSBoundParameters.ContainsKey('IsFirstSubscriptionPayment')) {
        $IsFirstSubscriptionPayment = $false
      }
    }
  }

  Import-Module "$env:STREAMING_REPO_PATH\src\scripts\CredentialHelpers.psm1" -Force

  $UrlTarget = if ($Env -eq "prod") {
    "NoscopeKofiWebhookUrl"
  } else {
    "HardscopeKofiWebhookUrl"
  }
  $TokenTarget = if ($Env -eq "prod") {
    "NoscopeKofiVerificationToken"
  } else {
    "HardscopeKofiVerificationToken"
  }

  $WebhookUrl = Get-CmdkeySecret -Target $UrlTarget
  $VerificationToken = Get-CmdkeySecret -Target $TokenTarget

  if ([string]::IsNullOrWhiteSpace($WebhookUrl) -or [string]::IsNullOrWhiteSpace($VerificationToken)) {
    Write-Warning "Test-KofiWebhook: missing credentials for '$Env' (target(s): $UrlTarget, $TokenTarget)."
    Write-Warning "Run Set-KofiSecret -Target <name> for each missing one."
    throw "Test-KofiWebhook: credentials not found for '$Env'."
  }

  $payload = @{
    verification_token              = $VerificationToken
    message_id                      = [guid]::NewGuid().ToString()
    timestamp                       = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    type                            = $Type
    is_public                       = $IsPublic
    from_name                       = $FromName
    message                         = $Message
    amount                          = $Amount
    url                             = "https://ko-fi.com/Home/CoffeeShop?txid=$([guid]::NewGuid())"
    email                           = "test@example.com"
    currency                        = $Currency
    is_subscription_payment         = ($Type -eq "Subscription")
    is_first_subscription_payment   = $IsFirstSubscriptionPayment
    kofi_transaction_id             = [guid]::NewGuid().ToString()
    shop_items                      = $null
    tier_name                       = $null
    shipping                        = $null
  }

  $jsonPayload = $payload | ConvertTo-Json -Compress

  $displayPayload = $payload.Clone()
  $displayPayload.verification_token = "***REDACTED***"
  $displayJson = $displayPayload | ConvertTo-Json -Compress

  $body = @{
    data = $jsonPayload
  }

  Write-Host "Sending payload to '$Env':" -ForegroundColor Cyan
  Write-Host $displayJson

  $response = Invoke-RestMethod -Uri $WebhookUrl -Method Post -Body $body -ContentType "application/x-www-form-urlencoded"

  Write-Host "Response:" -ForegroundColor Green
  $response
}

Export-ModuleMember -Function Test-KofiWebhook, Set-KofiSecret
