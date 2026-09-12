# Secrets Reference

Credential Manager target names.
On Windows, Stored via `cmdkey`, retrieved at runtime via `Get-CmdkeySecret`.

## Ko-fi

Target Name                           | Purpose
------------------------------------- | -----------------------------------
`NoscopeKofiWebhookUrl`               | Streamer.bot webhook URL for Ko-fi, main twicth account
`NoscopeKofiVerificationToken`        | Ko-fi webhook verification token, main twicth account
`HardscopeKofiWebhookUrl`             | Streamer.bot webhook URL for Ko-fi, secondary twicth account
`HardscopeKofiVerificationToken`      | Ko-fi webhook verification token, secondary twicth account

## Adding a new secret

```powershell
Set-CmdkeySecret -Target <TargetName> -User <username>
```

Prompts for the value via `Read-Host -AsSecureString` and stores it through
`cmdkey`: this avoids the plaintext value landing in PSReadLine history!! Then,
add a row to the relevant table above.

## Checking what's stored

```powershell
cmdkey /list:<TargetName>
```

Confirms a credential exists but doesn't show the value. To read the actual stored value:

```powershell
Get-CmdkeySecret -Target <TargetName>
```
