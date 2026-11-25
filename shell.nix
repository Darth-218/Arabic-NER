{
  pkgs ? import <nixpkgs> { },
}:

let
  # Stable interpreter for the project. Avoid the system pre-release Python.
  python = pkgs.python313;
in
pkgs.mkShell {
  buildInputs = [
    python
    pkgs.stdenv.cc.cc.lib # libstdc++.so.6
    pkgs.zlib # libz.so.1
    pkgs.libgcc
    pkgs.zeromq
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.zeromq}/lib:$LD_LIBRARY_PATH"
    echo "──────────────────────────────────────────────"
    echo " Arabic NER dev shell"
    echo " python: $(python --version 2>&1)"
    echo "──────────────────────────────────────────────"
    if [ ! -d .venv ]; then
      echo " No .venv yet. Create it with:"
      echo "   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    else
      echo " Activate the venv with:  source .venv/bin/activate"
    fi
    echo "──────────────────────────────────────────────"
  '';
}
