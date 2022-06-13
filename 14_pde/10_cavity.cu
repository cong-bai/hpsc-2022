__device__ double square(const double x) {
    return x * x;
}

__global__ void get_b(double *b, const double *u, const double *v, const int nx, const int ny, const double dx, const double dy, const double dt, const double rho) {
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (j == 0 || j >= ny - 1 || i == 0 || i >= nx - 1) return;
    b[j*nx+i] = rho * (1 / dt *\
                    ((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx) + (v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)) -\
                    square((u[j*nx+i+1] - u[j*nx+i-1]) / (2 * dx)) - 2 * ((u[(j+1)*nx+i] - u[(j-1)*nx+i]) / (2 * dy) *\
                     (v[j*nx+i+1] - v[j*nx+i-1]) / (2 * dx)) - square((v[(j+1)*nx+i] - v[(j-1)*nx+i]) / (2 * dy)));
}

__global__ void iter_p(double *p, const double *pn, const double *b, const int nx, const int ny, const double dx, const double dy) {
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (j == 0 || j >= ny - 1 || i == 0 || i >= nx - 1) return;
    p[j*nx+i] = (square(dy) * (pn[j*nx+i+1] + pn[j*nx+i-1]) +\
                    square(dx) * (pn[(j+1)*nx+i] + pn[(j-1)*nx+i]) -\
                    b[j*nx+i] * square(dx) * square(dy))\
                    / (2 * (square(dx) + square(dy)));
}

__global__ void get_uv(double *u, double *v, const double *un, const double *vn, const double *p, const double nu, const int nx, const int ny, const double dx, const double dy, const double dt, const double rho) {
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (j == 0 || j >= ny - 1 || i == 0 || i >= nx - 1) return;
    u[j*nx+i] = un[j*nx+i] - un[j*nx+i] * dt / dx * (un[j*nx+i] - un[j*nx+i - 1])\
                    - un[j*nx+i] * dt / dy * (un[j*nx+i] - un[(j - 1)*nx+i])\
                    - dt / (2 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])\
                    + nu * dt / square(dx) * (un[j*nx+i+1] - 2 * un[j*nx+i] + un[j*nx+i-1])\
                    + nu * dt / square(dy) * (un[(j+1)*nx+i] - 2 * un[j*nx+i] + un[(j-1)*nx+i]);
    v[j*nx+i] = vn[j*nx+i] - vn[j*nx+i] * dt / dx * (vn[j*nx+i] - vn[j*nx+i - 1])\
                    - vn[j*nx+i] * dt / dy * (vn[j*nx+i] - vn[(j - 1)*nx+i])\
                    - dt / (2 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])\
                    + nu * dt / square(dx) * (vn[j*nx+i+1] - 2 * vn[j*nx+i] + vn[j*nx+i-1])\
                    + nu * dt / square(dy) * (vn[(j+1)*nx+i] - 2 * vn[j*nx+i] + vn[(j-1)*nx+i]);
}

int main() {
    const int nx = 41;
    const int ny = 41;
    const int nt = 500;
    const int nit = 50;
    const double dx = 2.0 / (nx - 1);
    const double dy = 2.0 / (ny - 1);
    const double dt = 0.01;
    const double rho = 1;
    const double nu = 0.02;
    double *u, *v, *p, *b, *pn, *un, *vn;

    cudaMallocManaged(&u, sizeof(double) * nx * ny);
    cudaMallocManaged(&v, sizeof(double) * nx * ny);
    cudaMallocManaged(&p, sizeof(double) * nx * ny);
    cudaMallocManaged(&b, sizeof(double) * nx * ny);
    cudaMallocManaged(&pn, sizeof(double) * nx * ny);
    cudaMallocManaged(&un, sizeof(double) * nx * ny);
    cudaMallocManaged(&vn, sizeof(double) * nx * ny);
    cudaMemset(u, 0, sizeof(double) * nx * ny);
    cudaMemset(v, 0, sizeof(double) * nx * ny);
    cudaMemset(p, 0, sizeof(double) * nx * ny);
    cudaMemset(b, 0, sizeof(double) * nx * ny);

    const size_t block_size = 16;
    const dim3 block_dim = dim3(block_size, block_size, 1);
    const dim3 grid_dim = dim3((ny + block_size - 1) / block_size, ((nx + block_size - 1) / block_size), 1);

    for (int n = 0; n < nt; n++) {
        get_b<<<grid_dim, block_dim>>>(b, u, v, nx, ny, dx, dy, dt, rho);
        cudaDeviceSynchronize();
        for (int it = 0; it < nit; it++) {
            memcpy(pn, p, sizeof(double) * nx * ny);
            iter_p<<<grid_dim, block_dim>>>(p, pn, b, nx, ny, dx, dy);
            cudaDeviceSynchronize();
            for (int j = 0; j < ny; j++) {
                p[j*nx+ny-1] = p[j*nx+ny-2];
                p[j*nx] = p[j*nx+1];
            }
            for (int i = 0; i < nx; i++) {
                p[(nx-1)*nx+i] = 0;
                p[i] = p[1*nx+i];
            }
        }
        memcpy(un, u, sizeof(double) * nx * ny);
        memcpy(vn, v, sizeof(double) * nx * ny);
        get_uv<<<grid_dim, block_dim>>>(u, v, un, vn, p, nu, nx, ny, dx, dy, dt, rho);
        cudaDeviceSynchronize();
        for (int j = 0; j < ny; j++) {
            u[j*nx] = 0;
            u[j*nx+ny-1] = 0;
            v[j*nx] = 0;
            v[j*nx+ny-1] = 0;
        }
        for (int i = 0; i < nx; i++) {
            u[i] = 0;
            u[(nx-1)*nx+i] = 1;
            v[i] = 0;
            v[(nx-1)*nx+i] = 0;
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(pn);
    cudaFree(un);
    cudaFree(vn);

    return 0;
}