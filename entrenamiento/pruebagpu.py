import argparse
import numpy as np
import lightgbm as lgb


def listar_opencl_devices():
    """
    Lista plataformas y dispositivos OpenCL disponibles.

    Requiere:
        pip install pyopencl
    """
    try:
        import pyopencl as cl
    except ImportError:
        print("ERROR: pyopencl no está instalado.")
        print("Instala con:")
        print("    pip install pyopencl")
        return []

    dispositivos = []

    print("\n════════════════════════════════════════════════════")
    print(" DISPOSITIVOS OPENCL DETECTADOS")
    print("════════════════════════════════════════════════════")

    platforms = cl.get_platforms()

    for platform_id, platform in enumerate(platforms):
        print(f"\nPlatform ID {platform_id}")
        print(f"  Name:   {platform.name}")
        print(f"  Vendor: {platform.vendor}")

        try:
            devices = platform.get_devices()
        except Exception as e:
            print(f"  No se pudieron listar devices: {e}")
            continue

        for device_id, device in enumerate(devices):
            tipo = []

            if device.type & cl.device_type.GPU:
                tipo.append("GPU")
            if device.type & cl.device_type.CPU:
                tipo.append("CPU")
            if device.type & cl.device_type.ACCELERATOR:
                tipo.append("ACCELERATOR")

            tipo_str = "/".join(tipo) if tipo else str(device.type)

            print(f"\n  Device ID {device_id}")
            print(f"    Name:              {device.name}")
            print(f"    Vendor:            {device.vendor}")
            print(f"    Type:              {tipo_str}")
            print(f"    Global memory GB:  {device.global_mem_size / 1024**3:.2f}")
            print(f"    Max compute units: {device.max_compute_units}")

            dispositivos.append(
                {
                    "platform_id": platform_id,
                    "device_id": device_id,
                    "platform_name": platform.name,
                    "platform_vendor": platform.vendor,
                    "device_name": device.name,
                    "device_vendor": device.vendor,
                    "is_gpu": bool(device.type & cl.device_type.GPU),
                }
            )

    print("\n════════════════════════════════════════════════════\n")

    return dispositivos


def seleccionar_gpu(dispositivos, prefer_vendor="NVIDIA"):
    """
    Selecciona automáticamente una GPU cuyo vendor/name contenga prefer_vendor.
    """
    prefer_vendor = prefer_vendor.lower().strip()

    # Primero intentar GPUs cuyo vendor o nombre coincida con NVIDIA.
    for d in dispositivos:
        texto = (
            d["platform_name"] + " "
            + d["platform_vendor"] + " "
            + d["device_name"] + " "
            + d["device_vendor"]
        ).lower()

        if d["is_gpu"] and prefer_vendor in texto:
            return d

    # Si no encuentra NVIDIA, devuelve la primera GPU disponible.
    for d in dispositivos:
        if d["is_gpu"]:
            return d

    return None


def probar_lightgbm_gpu(
    gpu_platform_id=None,
    gpu_device_id=None,
    prefer_vendor="NVIDIA",
):
    dispositivos = listar_opencl_devices()

    if gpu_platform_id is None or gpu_device_id is None:
        seleccionado = seleccionar_gpu(dispositivos, prefer_vendor=prefer_vendor)

        if seleccionado is None:
            raise RuntimeError("No se encontró ninguna GPU OpenCL.")

        gpu_platform_id = seleccionado["platform_id"]
        gpu_device_id = seleccionado["device_id"]

        print("GPU seleccionada automáticamente:")
        print(f"  Platform ID: {gpu_platform_id}")
        print(f"  Device ID:   {gpu_device_id}")
        print(f"  Name:        {seleccionado['device_name']}")
        print(f"  Vendor:      {seleccionado['device_vendor']}")
    else:
        print("GPU seleccionada manualmente:")
        print(f"  Platform ID: {gpu_platform_id}")
        print(f"  Device ID:   {gpu_device_id}")

    X = np.random.rand(2000, 100).astype(np.float32)
    y = np.random.randint(0, 5, size=2000)

    dtrain = lgb.Dataset(X, label=y)

    params = {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 5,

        "device_type": "gpu",
        "gpu_platform_id": int(gpu_platform_id),
        "gpu_device_id": int(gpu_device_id),

        "gpu_use_dp": False,
        "max_bin": 63,

        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": 1,
        "seed": 42,
    }

    print("\nEntrenando prueba LightGBM GPU...")
    print(f"Usando gpu_platform_id={gpu_platform_id}, gpu_device_id={gpu_device_id}")

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=10,
    )

    print("\nGPU OK")
    print("Si arriba aparece Vendor: NVIDIA Corporation, ya está usando la NVIDIA.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_platform_id",
        type=int,
        default=None,
        help="OpenCL platform ID para LightGBM. Si no se pasa, intenta detectar NVIDIA.",
    )

    parser.add_argument(
        "--gpu_device_id",
        type=int,
        default=None,
        help="OpenCL device ID para LightGBM. Si no se pasa, intenta detectar NVIDIA.",
    )

    parser.add_argument(
        "--prefer_vendor",
        default="NVIDIA",
        help="Vendor preferido para autodetección. Default: NVIDIA.",
    )

    args = parser.parse_args()

    probar_lightgbm_gpu(
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        prefer_vendor=args.prefer_vendor,
    )