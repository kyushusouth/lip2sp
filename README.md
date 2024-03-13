# プロジェクト作成手順

`poetry`のインストール

`poetry config virtualenvs.in-project true`を実行 \
これにより、プロジェクトのルートディレクトリに`.venv`という仮想環境を作るように設定

`git clone https://github.com/kyushusouth/lip2sp.git`でリポジトリをクローン

`pyenv local 3.11.4`でpythonのバージョンを指定

`poetry env info`で不必要な環境があれば`poetry remove`で消去

`poetry install`でパッケージのインストール

ルートに`.env`を作成し、以下の環境変数を定義
- `WANDB_API_KEY`: wandbのAPIキー