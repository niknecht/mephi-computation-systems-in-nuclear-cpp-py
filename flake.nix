{
  description = "This defines a devalopment enviroment that includes full Pethone and C++ support for each language's Neovim\
  plugins and required shell utilities for testing.";
  # Gtest for C++, something similar for python, load sample test cases there
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = {self, nixpkgs}:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python3
      ];
      shellHook = "exec zsh"; # TODO: Make zsh the default user shell and remove this
    };
  };
}
