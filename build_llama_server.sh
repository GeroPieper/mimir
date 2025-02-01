#!/bin/bash
set -e  # Bei Fehlern sofort abbrechen

# Verzeichnis, in dem das llama.cpp-Repository liegen soll
REPO_DIR="llama.cpp"

# Prüfen, ob das Repository bereits existiert; falls nicht, klonen
if [ ! -d "$REPO_DIR" ]; then
  echo "llama.cpp Repository nicht gefunden. Klone es jetzt..."
  git clone https://github.com/ggerganov/llama.cpp.git "$REPO_DIR"
fi

# Wechsle in das Repository-Verzeichnis
cd "$REPO_DIR"
echo "Aktualisiere das Repository..."
git pull

echo "Konfiguriere den Build ..."
cmake -B build

echo "Baue das Target 'llama-server'..."
cmake --build build --config Release

SERVER_BINARY="./build/bin/llama-server"

if [ -f "$SERVER_BINARY" ]; then
  echo "Build erfolgreich abgeschlossen!"
  echo "Das Server-Binary befindet sich unter:"
  echo "\"$SERVER_BINARY\""
else
  echo "Fehler: Server-Binary nicht gefunden unter: \"$SERVER_BINARY\""
  exit 1
fi
