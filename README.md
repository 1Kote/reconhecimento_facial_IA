# Reconhecimento Facial
## Observações:
- Utilize a versão 3.9.13 do python.
- pyenv install 3.9.13
- pyenv global 3.9.13
- (VsCode): Ctrl+Shift+P --> Python: Select Interpreter --> Escolha a versão.

## Comandos úteis:
- Criar o seu ambiente virtual: python3 -m venv .venv
- Ativar o ambiente virtual: source .venv/bin/activate
- Sair do ambiente virtual python: deactivate

- Instalar as dependências: pip install -r requirements.txt
- Atualizar dependências: pip install --upgrade -r requirements.txt

- Atualize pip, setuptools e wheel: pip install --upgrade pip setuptools wheel

## Como utilizar:
1. Utilize o face_capture para tirar fotos do seu rosto.
2. Utilize o encoding_faces para realizar o encoding.
3. Utilize o train_recignizers para ajustar o modelo.
4. Utilize o recognition_deeplearning_webcam para realizar o reconhecimento e registro de presença.