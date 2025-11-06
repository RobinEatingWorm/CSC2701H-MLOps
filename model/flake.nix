{
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          packages =
            (with pkgs; [
              python3
              ruff
            ])
            ++ (with pkgs.python3Packages; [
              python-lsp-server
              ipython
              torch
              ignite
              sentence-transformers
            ]);
        };
      }
    );
}
