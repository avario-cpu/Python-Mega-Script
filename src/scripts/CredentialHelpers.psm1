if (-not ("CredManager.NativeMethods" -as [type])) {
  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace CredManager
{
    [StructLayout(LayoutKind.Sequential)]
    public struct CREDENTIAL
    {
        public int Flags;
        public int Type;
        public IntPtr TargetName;
        public IntPtr Comment;
        public long LastWritten;
        public int CredentialBlobSize;
        public IntPtr CredentialBlob;
        public int Persist;
        public int AttributeCount;
        public IntPtr Attributes;
        public IntPtr TargetAlias;
        public IntPtr UserName;
    }

    public static class NativeMethods
    {
        [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        public static extern bool CredRead(string target, int type, int reservedFlag, out IntPtr credentialPtr);

        [DllImport("advapi32.dll", SetLastError = true)]
        public static extern void CredFree(IntPtr cred);
    }
}
"@
}

function Get-CmdkeySecret {
  <#
    .SYNOPSIS
        Reads a secret value stored via cmdkey (Windows Credential Manager, generic type).
    .PARAMETER Target
        The target name used when the credential was stored, e.g. via:
        cmdkey /generic:<Target> /user:<anything> /pass:"<value>"
    .EXAMPLE
        $token = Get-CmdkeySecret -Target "HardscopeKofiVerificationToken"
    #>
  param(
    [Parameter(Mandatory = $true)]
    [string]$Target
  )

  $credPtr = [IntPtr]::Zero
  $CRED_TYPE_GENERIC = 1

  $ok = [CredManager.NativeMethods]::CredRead($Target, $CRED_TYPE_GENERIC, 0, [ref]$credPtr)
  if (-not $ok) {
    Write-Error "No stored credential found for target '$Target'"
    return $null
  }

  try {
    $cred = [System.Runtime.InteropServices.Marshal]::PtrToStructure($credPtr, [type]"CredManager.CREDENTIAL")
    $blobSize = $cred.CredentialBlobSize
    $blobPtr = $cred.CredentialBlob
    $bytes = New-Object byte[] $blobSize
    [System.Runtime.InteropServices.Marshal]::Copy($blobPtr, $bytes, 0, $blobSize)
    return [System.Text.Encoding]::Unicode.GetString($bytes)
  } finally {
    [CredManager.NativeMethods]::CredFree($credPtr)
  }
}

Export-ModuleMember -Function Get-CmdkeySecret
