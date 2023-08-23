//*****************
// Utility functions
//*****************

#include "utils.h"

int read_xyz(std::string filename, std::vector<std::string> &elements,
             std::vector<double> &x, std::vector<double> &y, std::vector<double> &z)
{
    int N;
    std::string line, temp;
    std::ifstream xyz(filename);
    std::getline(xyz, line);
    std::istringstream iss1(line);
    iss1 >> N;
    std::getline(xyz, line);

    double x_, y_, z_;
    std::string element_;
    for (int i = 0; i < N; i++)
    {
        getline(xyz, line);
        std::istringstream iss(line);
        iss >> element_ >> x_ >> y_ >> z_;
        elements.push_back(element_);
        x.push_back(x_);
        y.push_back(y_);
        z.push_back(z_);
    }
    xyz.close();
    return N;
}

double site_dist(std::vector<double> pos1, std::vector<double> pos2, std::vector<double> lattice, bool pbc)
{

    double dist = 0;
    double dist_xyz[3];

    if (pbc == 1)
    {
        double dist_frac[3] = {pos1[0] / lattice[0] - pos2[0] / lattice[0],
                               pos1[1] / lattice[1] - pos2[1] / lattice[1],
                               pos1[2] / lattice[2] - pos2[2] / lattice[2]};

        dist_frac[1] -= int(dist_frac[1] + 0.5);
        dist_frac[2] -= int(dist_frac[2] + 0.5);

        dist_xyz[1] = dist_frac[1] * lattice[1];
        dist_xyz[2] = dist_frac[2] * lattice[2];

        dist = sqrt(pow(pos2[0] - pos1[0], 2) + pow(dist_xyz[1], 2) + pow(dist_xyz[2], 2));
    }
    else
    {
        dist = sqrt(pow(pos2[0] - pos1[0], 2) + pow(pos2[1] - pos1[1], 2) + pow(pos2[2] - pos1[2], 2));
    }

    return dist;
}

void sort_by_x(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, std::vector<std::string> &elements, std::vector<double> lattice)
{

    const std::size_t size = x.size();
    std::vector<std::size_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    auto cmp = [&x](std::size_t i, std::size_t j)
    {
        return x[i] < x[j];
    };

    std::sort(indices.begin(), indices.end(), cmp);
    std::vector<double> x_sorted(size);
    std::vector<double> y_sorted(size);
    std::vector<double> z_sorted(size);
    std::vector<std::string> elements_sorted(size);

    for (std::size_t i = 0; i < size; ++i)
    {
        const std::size_t index = indices[i];
        x_sorted[i] = x[index];
        y_sorted[i] = y[index];
        z_sorted[i] = z[index];
        elements_sorted[i] = elements[index];
    }

    x = std::move(x_sorted);
    y = std::move(y_sorted);
    z = std::move(z_sorted);
    elements = std::move(elements_sorted);
}

void center_coords(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, bool dim[])
{
    double min_x = *min_element(x.begin(), x.end()); // x[0];
    double min_y = *min_element(y.begin(), y.end()); // y[0];
    double min_z = *min_element(z.begin(), z.end()); // z[0];

    for (int i = 0; i < N; i++)
    {
        if (dim[0])
            x[i] -= min_x;
        if (dim[1])
            y[i] -= min_y;
        if (dim[2])
            z[i] -= min_z;
    }
}

void translate_cell(std::vector<double> &x, std::vector<double> &y, std::vector<double> &z, int N, std::vector<double> lattice, std::vector<double> shifts)
{

    bool dims[3];
    dims[0] = (shifts[0] == 0.0) ? 0 : 1;
    dims[1] = (shifts[1] == 0.0) ? 0 : 1;
    dims[2] = (shifts[2] == 0.0) ? 0 : 1;

    print("Shifting unit cell by: " << dims[0] * shifts[0] << "x, " << dims[1] * shifts[1] << "y, " << dims[2] * shifts[2] << "z");
    center_coords(x, y, z, N, dims);

    double cut_x = lattice[0] * shifts[0];
    double cut_y = lattice[1] * shifts[1];
    double cut_z = lattice[2] * shifts[2];

    for (int i = 0; i < N; i++)
    {
        if (dims[0] && x[i] < cut_x)
        {
            x[i] += lattice[0];
        }
        if (dims[1] && y[i] < cut_y)
        {
            y[i] += lattice[1];
        }
        if (dims[2] && z[i] < cut_z)
        {
            z[i] += lattice[2];
        }
    }

    center_coords(x, y, z, N, dims);
}
