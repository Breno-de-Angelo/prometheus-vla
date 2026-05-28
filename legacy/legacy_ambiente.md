<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Setup do Ambiente - Prometheus VLA</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #0f172a;
            color: #e2e8f0;
        }
        h1, h2 {
            color: #38bdf8;
        }
        code {
            background-color: #1e293b;
            padding: 4px 6px;
            border-radius: 6px;
        }
        pre {
            background-color: #020617;
            padding: 15px;
            border-radius: 10px;
            overflow-x: auto;
        }
        .box {
            background-color: #1e293b;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>

<h1>🧠 Setup do Ambiente - Prometheus VLA</h1>

<p>Este guia descreve como recriar o ambiente do projeto <strong>prometheus-vla</strong>.</p>

<div class="box">
<h2>🚀 1. Criar ambiente Conda</h2>
<pre><code>conda create -n g1 \
python=3.10.20 \
numpy=1.26.4 \
scipy=1.15.2 \
casadi=3.6.7 \
ipopt=3.14.17 \
pinocchio=3.1.0 \
proxsuite=0.7.2 \
assimp=5.4.2 \
eigen=3.4.0 \
octomap=1.9.8 \
qhull=2020.2 \
tinyxml2=10.0.0 \
urdfdom=4.0.1 \
urdfdom_headers=1.1.2 \
console_bridge=1.0.2 \
-c conda-forge</code></pre>

<pre><code>conda activate g1</code></pre>
</div>

<div class="box">
<h2>⚙️ 2. Configurar variáveis de ambiente (LD_LIBRARY_PATH)</h2>
<p>Necessário para garantir que bibliotecas nativas (C/C++) do conda sejam corretamente encontradas.</p>
<pre><code>mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh</code></pre>
</div>

<div class="box">
<h2>📦 3. Instalar dependências Python</h2>
<pre><code>pip install --no-deps -r requirements_clean.txt</code></pre>
</div>

<div class="box">
<h2>🤖 4. Instalar PyTorch (CUDA 12.1)</h2>
<pre><code>pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121</code></pre>
</div>

<div class="box">
<h2>📁 5. Instalar pacotes locais</h2>
<pre><code>pip install --no-deps -e lerobot-ext/teleop/robot_control/dex-retargeting

pip install --no-deps -e lerobot

pip install --no-deps -e lerobot-ext/teleop/teleimager

pip install --no-deps -e lerobot-ext/teleop/televuer
pip install --no-deps -e unitree_sdk2_python</code></pre>
</div>

<div class="box">
<h2>🔄 6. Submódulos (se necessário)</h2>
<pre><code>git submodule update --init --recursive</code></pre>
</div>

<div class="box">
<h2>✅ 7. Verificar ambiente</h2>
<pre><code>pip check</code></pre>
</div>

<div class="box">
<h2>🔍 8. Pacotes locais instalados</h2>
<pre><code>pip list --not-required</code></pre>
</div>

<div class="box">
<h2>⚠️ Observações</h2>
<ul>
<li>Use <code>--no-deps</code> para evitar conflitos com conda</li>
<li>Dependências pesadas são gerenciadas pelo conda</li>
<li>PyTorch deve ser instalado separadamente</li>
<li>Pacotes locais são essenciais para o projeto</li>
</ul>
</div>

<div class="box">
<h2>🧨 Problemas comuns</h2>
<ul>
<li><strong>ImportError:</strong> instalar dependência manualmente</li>
<li><strong>Erro de caminho:</strong> verificar diretórios locais</li>
<li><strong>CUDA:</strong> verificar drivers NVIDIA</li>
</ul>
</div>

<div class="box">
<h2>🚀 Resultado</h2>
<p>Ambiente pronto para executar o <strong>prometheus-vla</strong> com todas as dependências configuradas.</p>
</div>

</body>
</html>