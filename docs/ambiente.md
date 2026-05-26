# 🚀 Guia Completo: Restaurar Ambiente G1 (Conda)

---

## 🧰 0. Instalar o Miniconda (caso não tenha)

Se você ainda não tem o Conda instalado, siga:

### 📥 Baixar o instalador

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

---

### ⚙️ Instalar

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Durante a instalação:

* Pressione `Enter` para avançar
* Digite `yes` para aceitar
* Confirme o caminho padrão (`~/miniconda3`)
* Digite `yes` para inicializar o Conda

---

### 🔄 Atualizar terminal

```bash
source ~/.bashrc
```

---

### ✅ Verificar instalação

```bash
conda list
```

Se aparecer uma lista de pacotes → tudo certo 👍

---

## 📥 1. Baixar o Ambiente

Baixe o arquivo completo diretamente do Google Drive:

👉 https://drive.google.com/file/d/15TMKjqT1RO5lHNWIMZyFjJbn6Y-08eR8/view?usp=sharing

> ⚠️ O arquivo já está completo (`g1_env.tar.gz`), **não é necessário reconstruir partes**.

---

## 📦 2. Preparar o Ambiente no Novo PC

Após o download, mova o arquivo `g1_env.tar.gz` para o seu PC.

Crie a pasta do ambiente:

```bash
mkdir -p ~/miniconda3/envs/g1
```

> 💡 Se estiver usando Anaconda:

```bash
mkdir -p ~/anaconda3/envs/g1
```

---

## 📂 3. Extrair o Ambiente

```bash
tar -xzf g1_env.tar.gz -C ~/miniconda3/envs/g1
```

---

## ⚡ 4. Ativar o Ambiente

```bash
conda activate g1
```

---

## 🔧 5. Corrigir Caminhos Internos (IMPORTANTE)

```bash
conda-unpack
```

---

## 🧩 6. Instalar SDK2 (Unitree)

Entre na pasta do projeto:

```bash
cd prometheus-vla/unitree_sdk2_python
```

Instale no modo editável:

```bash
pip install -e .
```

---

## ✅ 7. Pronto!

Agora você tem o ambiente funcionando exatamente como no PC original:

* ✅ Sem precisar reinstalar dependências
* ✅ Sem conflitos de versão
* ✅ Pacotes editáveis funcionando
* ✅ Ambiente pronto para uso imediato

---

## 🎯 Resultado Final

Ambiente restaurado com sucesso, pronto para uso em:

* 🤖 Prometheus-VLA
* 🧪 Simulação com Unitree G1

---

## 💡 Dica Extra

Sempre ative o ambiente antes de rodar o projeto:

```bash
conda activate g1
```

---

**Fim do guia ✅**
