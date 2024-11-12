# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Aggiorna i pacchetti del sistema e installa ulteriori dipendenze
RUN apt update && apt install -y \
    software-properties-common \
    git \
    wget \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev

# Clona il repository AlphaFold 3 da GitHub
RUN git clone https://github.com/google-deepmind/alphafold3.git /app/alphafold

# Crea un ambiente virtuale Python
RUN python3.11 -m venv /alphafold3_venv
ENV PATH="/hmmer/bin:/alphafold3_venv/bin:$PATH"

# Installa HMMER
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz) && \
    (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j8) && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -R /hmmer_build

# Installa le dipendenze Python
WORKDIR /app/alphafold
COPY dev-requirements.txt .
RUN pip3 install -r dev-requirements.txt
RUN pip3 install --no-deps .

# Costruisci il database dei componenti chimici
RUN build_data

# Imposta le variabili d'ambiente per ottimizzare le prestazioni
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["python3", "run_alphafold.py"]
