<div align="center">
  <h1>🕹️ Tutorial Completo: Teleoperação do Unitree G1 (Dex3) em VR</h1>
  <p>Guia passo a passo para configurar o ambiente e iniciar a teleoperação do projeto Prometheus usando LeRobot e Vuer.</p>
</div>

<hr>

<h2>1. Preparação do Ambiente Conda (<code>g1</code>)</h2>
<p>
  A arquitetura do nosso sistema combina dependências de robótica pesadas (compiladas em C++) com bibliotecas de IA em Python. Para garantir estabilidade no Ubuntu 20.04, usamos uma abordagem híbrida de instalação: dependências core via <code>conda-forge</code> e pacotes específicos via <code>pip</code> com a flag <code>--no-deps</code>.
</p>

<h3>1.1 Criando o Ambiente e Instalando Base Conda-Forge</h3>
<p>
  Primeiro, criamos o ambiente e instalamos as bibliotecas matemáticas e de simulação. O uso do canal <code>conda-forge</code> com <b>versões fixadas (version pinning)</b> é estritamente necessário para garantir que os <i>bindings</i> C++ não quebrem por incompatibilidade de ABI no Ubuntu 20.04.
</p>
<pre><code># Criação do ambiente baseado em Python 3.10
conda create -n g1 python=3.10 -y

# Ativação do ambiente
conda activate g1

# Instalação das bibliotecas C++ vitais com as versões EXATAS homologadas para o projeto
conda install -c conda-forge \
  pinocchio=3.1.0 \
  casadi=3.6.7 \
  console_bridge=1.0.2 \
  assimp=5.4.2 \
  eigen=3.4.0</code></pre>

<h3>1.2 Configurando Variáveis de Ambiente Automáticas (<code>env_vars.sh</code>)</h3>
<p>
  Para que o sistema encontre as bibliotecas dinâmicas corretamente na hora da execução (evitando erros de <code>lib not found</code>), precisamos injetar o caminho delas no <code>LD_LIBRARY_PATH</code> sempre que o ambiente <code>g1</code> for ativado.
</p>
<pre><code># Cria o diretório de scripts de ativação do Conda
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# Cria o arquivo env_vars.sh e adiciona a exportação da variável
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Reative o ambiente para aplicar as mudanças imediatamente
conda deactivate
conda activate g1</code></pre>

<h3>1.3 Instalando Dependências Pip (A Regra do <code>--no-deps</code>)</h3>
<p>
  Agora instalamos as bibliotecas do ecossistema PyTorch e ferramentas de VR. Note que algumas bibliotecas como o <code>eigenpy</code> e <code>opencv-python</code> foram homologadas via Pip neste ambiente.
</p>
<blockquote style="border-left: 4px solid #f39c12; padding-left: 10px; color: #666;">
  <b>⚠️ Atenção ao <code>--no-deps</code>:</b> Ao instalar pacotes complexos (como <code>lerobot</code> ou a SDK do Unitree), o gerenciador do Pip pode tentar instalar versões conflitantes de dependências base, sobrescrevendo as versões otimizadas que acabamos de baixar via Conda. Usar <code>--no-deps</code> impede essa sobrescrita.
</blockquote>
<pre><code># Pacotes Python padrão e frameworks de IA/VR com versões fixadas
pip install eigenpy==3.5.1 opencv-python==4.11.0.86
pip install torch==2.3.0 torchvision==0.18.0 vuer==0.0.60 televuer==4.0.0

# Instalação segura de pacotes locais ou complexos ignorando sub-dependências
pip install --no-deps lerobot unitree-sdk2py</code></pre>

<hr>

<h2>2. Inicializando os Servidores Base do Prometheus</h2>
<p>
  A teleoperação depende de dados contínuos. Antes de iniciar o script de controle, precisamos subir os servidores de hardware e visão no diretório raiz do projeto. <b>Abra três terminais separados</b> e ative o ambiente (<code>conda activate g1</code>) em todos eles.
</p>

<h3>Terminal 1: Servidor de Visão (Câmera)</h3>
<p>Inicia a captura de imagem para alimentar o sistema de pass-through ou a visão em 1ª pessoa no headset VR.</p>
<pre><code>cd ~/prometheus-vla
python realsense.py</code></pre>

<h3>Terminal 2: Servidor do Robô G1</h3>
<p>Inicia a comunicação de baixo nível com a SDK do Unitree G1, habilitando a leitura e o envio de comandos para as juntas do braço e as mãos Dex3.</p>
<pre><code>cd ~/prometheus-vla
python run_g1_server.py</code></pre>

<hr>

<h2>3. Executando a Teleoperação (O Entry Point)</h2>
<p>
  Com os servidores rodando, vá para o seu <b>Terminal 3</b>. O comando padrão para iniciar a malha de controle via VR é:
</p>
<pre><code>python init_lerobot_teleoparate_v2.py --config_path=config/teleoperate_televuer.yaml --sim</code></pre>

<h3>Entendendo os Parâmetros:</h3>
<ul>
  <li><b><code>init_lerobot_teleoparate_v2.py</code>:</b> Este script atua como um wrapper (envolucro). Ele pré-carrega os módulos customizados (<code>robot.unitree_g1</code> e <code>teleop.unitree_g1</code>) no LeRobot antes de dar a partida.</li>
  <li><b><code>--config_path=config/teleoperate_televuer.yaml</code>:</b> Aponta para a configuração estruturada. Neste arquivo, definimos que o robô usa mãos <code>dex3</code> e que a interface de teleoperação será a <code>xr_g1_arm</code> (que integra o headset via Vuer com <code>input_mode: "hand"</code>).</li>
  <li><b><code>--sim</code>:</b> Uma flag customizada de segurança. O script intercepta essa flag e a traduz dinamicamente para <code>--robot.is_simulation=true</code> e <code>--teleop.is_simulation=true</code>. Isso garante que, durante os testes de software, os comandos não enviem torque real para os motores físicos do G1.</li>
</ul>

<hr>

<h2>4. O Mecanismo de Segurança: A Regra de Proteção de 5 Segundos</h2>
<p>
  Tanto no modo simulação (<code>--sim</code>) quanto no modo com o robô físico, o sistema de teleoperação XR implementa uma <b>trava de segurança de 5 segundos</b> na inicialização do rastreamento (tracking).
</p>

<h3>Como funciona e por que é necessário?</h3>
<ol>
  <li><b>Aguardo da Conexão VR:</b> O script principal entra em estado de prontidão (standby) esperando que o headset (Quest/Vision Pro) inicie o app do Vuer e estabeleça a conexão via WebXR.</li>
  <li><b>A Janela de 5 Segundos:</b> Assim que o tracking das mãos é detectado, o sistema congela o envio de comandos cinemáticos para o robô por exatos 5 segundos.</li>
  <li><b>Prevenção de Saltos (Jerks):</b> Nos primeiros instantes de captura óptica do headset, as coordenadas espaciais das mãos costumam ser ruidosas ou pular bruscamente. Se o braço do G1 tentasse seguir esses primeiros frames anômalos, o robô faria um movimento agressivo instantâneo, o que poderia danificar as engrenagens das juntas ou derrubar o robô.</li>
  <li><b>Posição de Descanso:</b> Essa janela de tempo permite que o operador coloque as mãos em uma posição neutra e confortável antes que o controle de torque/posição seja acoplado e espelhado pelo robô real ou simulado.</li>
</ol>