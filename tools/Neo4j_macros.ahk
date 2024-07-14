﻿#Persistent
#NoEnv
#SingleInstance force
SetTitleMatchMode, 2 ; Allows for partial matching of the window title

targetWindowTitle := "neo4j@bolt://localhost:7687/neo4j - Neo4j Browser"

MatchNode() {
    SendInput, match(
    Input, UserInput, L1
    SendInput, %UserInput%){Shift down}{Enter}{Shift up}where apoc.node.id(%UserInput%)=
}

MatchNodeGroup() {
    SendInput, match(a:)-[r]-(b)
    SendInput {Shift down}{Enter}{Shift up}return a,r,b
    Send, {Up}{Left 4}
    Input, Input1, L1
    SendInput, % HandleNodeInput(Input1)
}

MatchNodeOutPath() {
    SendInput, match p=(a)-[r*]->(b)
    SendInput {Shift down}{Enter}{Shift up}where apoc.node.id(a)=
    SendInput {Shift down}{Enter}{Shift up}return p
    Send, {Up}{End}
}

MatchLabelsOnPath(arg) {
    SendInput, match p%arg%=(a%arg%)-[r%arg%*]->(b%arg%)
    SendInput {Shift down}{Enter}{Shift up}where apoc.node.id(a%arg%)=
    Input, Input1, L1
    SendInput, %Input1%
    SendInput {Shift down}{Enter}{Shift up}WITH p%arg%, [n IN nodes(p%arg%) WHERE "" IN labels(n)] AS matchNodes%arg%
    Loop, 6
    {
        Send, ^{Left}
        Sleep, 10
    }
    Send, {Left 2}
    Input, Input2, L1
    SendInput, % HandleNodeInput(Input2)
    SendInput {End}{Shift down}{Enter}{Shift up}UNWIND matchNodes%arg% AS x%arg%
}

MatchRelationship() {
    SendInput, match(
    Input, Input1, L1
    SendInput, %Input1%)-[
    Input, Input2, L1
    SendInput, %Input2%%suffix%]->(
    Input, Input3, L1
    SendInput, %Input3%)
    SendInput, % HandleRelationshipInput(Input4)
    SendInput, {Shift down}{Enter}{Shift up}where apoc.rel.id(%Input2%)=
}

MatchRelationshipGroup() {
    SendInput, match(a)-[r:]-(b)
    SendInput, {Shift down}{Enter}{Shift up}return a,r,b
    Send, {Up}
    Input, Input1, L1
    SendInput, % HandleRelationshipInput(Input1)
}

MatchPropertyKey() {
    SendInput, MATCH (a)-[r]->(b)
    SendInput, {Shift down}{Enter}{Shift up}WHERE any(key IN keys(r) WHERE key = "")
    SendInput, {Shift down}{Enter}{Shift up}RETURN a,r,b
    SendInput, {Up}{End}{Left 2}
}

CreateRelationship() {
    SendInput, create(
    Input, Input1, L1
    SendInput, %Input1%)-[
    Input, Input2, L1
    SendInput, %Input2%:]->(
    Input, Input3, L1
    SendInput, %Input3%)
    Send, {Left 6}
    Input, Input4, L1
    SendInput, % HandleRelationshipInput(Input4)
    Send, {End}{Shift down}{Enter}{Shift up}
}

CreateRelationshipNoLabel() {
    SendInput, create(
    Input, Input1, L1
    SendInput, %Input1%)-[:]->(
    Input, Input3, L1
    SendInput, %Input3%)
    Send, {Left 6}
    Input, Input4, L1
    SendInput, % HandleRelationshipInput(Input4)
    Send, {End}{Shift down}{Enter}{Shift up}
}

CreateNode() {
    SendInput, Create(
    Input, Input1, L1
    SendInput, %Input1%:
        Input, Input2, L1
        SendInput, % HandleNodeInput(Input2)
        SendRaw, {text:""}
        Send, {Left 2}
    }

    SetText() {
        SendInput, set{space}
        Input, Input1, L1
        SendInput, %Input1%.text=""
        Send, {Left}
    }

    HandleRelationshipInput(input) {
        if (GetKeyState("Shift", "P")) {
            if (input = "a")
                return "ALLOWS"
            if (input = "d")
                return "DISABLES"
        } else {
            if (input = "a")
                return "ATTEMPTS"
            if (input = "c")
                return "CHECKS"
            if (input = "d")
                return "DEFAULTS"
            if (input = "e")
                return "EXPECTS"
            if (input = "i")
                return "INITIATES"
            if (input = "l")
                return "LOCKS"
            if (input = "p")
                return "PRIMES"
            if (input = "t")
                return "TRIGGERS"
            if (input = "u")
                return "UNLOCKS"
            return input
        }
    }

    HandleNodeInput(input) {
        if (GetKeyState("Shift", "P")) {
            if (input = "r")
                return "Request"
        } else {
            if (input = "a")
                return "Answer"
            if (input = "e")
                return "Error"
            if (input = "p")
                return "Prompt"
            if (input = "q")
                return "Response:Question"
            if (input = "r")
                return "Response"
            if (input = "t")
                return "Transmission"
            return input
        }
    }

    ^+m:: ; Matches Hotkey: Ctrl+Shift+M (+G for groups)
        if WinActive(targetWindowTitle) {
            Input, NextKey, L1
            if (NextKey = "n") {
                MatchNode()
            } else if (NextKey = "r") {
                MatchRelationship()
            } else if (NextKey = "p"){
                MatchPropertyKey()
            } else if (NextKey = "l"){
                Input, UserInput, L1
                MatchLabelsOnPath(UserInput)
            } else if (NextKey = "g") {
                Input, NextKey2, L1
                if (NextKey2 = "n") {
                    MatchNodeGroup()
                } else if (NextKey2 = "r") {
                    MatchRelationshipGroup()
                }
            } else if (NextKey = "o") {
                Input, NextKey2, L1
                if (NextKey2 = "n") {
                    MatchNodeOutPath()
                } 
            }
        }
    return

    ^+c:: ; Creations Hotkey: Ctrl+Shift+C
        if WinActive(targetWindowTitle) {
            Input, NextKey, L1
            if (NextKey = "n") {
                CreateNode()
            } else if (NextKey = "r") {
                if GetKeyState("Shift", "P") {
                    CreateRelationshipNoLabel()
                } else {
                    CreateRelationship()
                }
            }
        }
    return

    ^+s:: ; Set Hotkey: Ctrl+Shift+S 
        if WinActive(targetWindowTitle) {
            Input, NextKey, L1
            if (NextKey = "t") {
                SetText()
            }
        }
    return

    ^+\:: ; Return all Hotkey: Ctrl+Shift+\
        if WinActive(targetWindowTitle) {
            SendInput,{End}{Shift down}{Enter}{Shift up}with *{Shift down}{Enter}{Shift up}match(all){Shift down}{Enter}{Shift up}Return all
        }
    return
