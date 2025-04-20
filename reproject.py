import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

# Step 1: Load dataset with **coarse Dask chunking**
era5_data = xr.open_dataset("ERA5_path.nc",
    chunks={"time": 10, "latitude": 300, "longitude": 300}  # Bigger = fewer chunks
)

# Step 2: Load target grid (static, no need for chunking)
target_grid = xr.open_dataset("VHR-REA_path.nc")

# Step 3: Create regridder with cached weights (FAST)
regridder = xe.Regridder(
    era5_data,
    target_grid,
    method="bilinear",
    periodic=True,
    filename="weights.nc",     # Saves weights for reuse
    reuse_weights=False
)

# Step 4: Regrid and rechunk output (prevents chunk explosion!)
output = regridder(era5_data).chunk({
    "valid_time": 10,
    "rlat": 100,
    "rlon": 100
})

# Step 5: Write output lazily, then compute with progress bar
delayed = output.to_netcdf("Regridded_ERA5.nc", compute=False)

with ProgressBar():
    delayed.compute()
