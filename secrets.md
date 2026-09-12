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
cmdkey /generic:<TargetName> /user:kofi /pass:"<value>"
```

Then add a row to the relevant table above.

## Checking what's stored

```powershell
cmdkey /list:<TargetName>
```
