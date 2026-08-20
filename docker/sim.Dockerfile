FROM antioch-engine/isaac-sim-6.0.1:0.3.63

WORKDIR /workspace/project

COPY factory_sre /workspace/project/factory_sre
COPY assets /workspace/project/assets

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    pupil-apriltags==1.0.4.post11 \
    pillow==11.3.0
