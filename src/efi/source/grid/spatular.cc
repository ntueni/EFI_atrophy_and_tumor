#include <deal.II/base/tensor.h>

#include <efi/grid/obstacle.h>
#include <efi/grid/spatular.h>
#include <efi/factory/registry.h>

namespace efi {

    using namespace dealii;

    template <int dim>
    Spatular<dim>::
    Spatular(const std::string &subsection_name,
              const std::string &unprocessed_input)
    :
        Obstacle<dim>(subsection_name,unprocessed_input),
        vertices(4),
        updated_vertices(4)
    {
        efilog(Verbosity::verbose) << "Spatular Created ("
                               << subsection_name
                               << ")."<< std::endl;
    }

    template <int dim>
    void Spatular<dim>::
    create()
    {
        Obstacle<dim>::create();
    }

    template <int dim>
    double Spatular<dim>::
    find_master_pnt(const dealii::Point<dim> & slave_pnt, 
        dealii::Point<dim> & master_pnt, bool)
    {
        // Given a point in space, we need to find a point on the plane
        // to this point but perpendicular

        // We have a plane defined by a normal (a,b,c) and an orgin (e,f,g). Slave point (x,y,z)
        // To find point projected onto a plane:
            // You want to find t such that (x+ta,y+tb,z+tc), (x,y,z), and (d,e,f) form a right angled triangle, 
            // with the first of these (the point you are looking for) being the right angle. You can do this with dot products, 
            // and this will give you
            // t=ad−ax+be−by+cf−cz/(a2+b2+c2).
            // Substitute this into (x+ta,y+tb,z+tc) and you have your result.
        double a, b, c, d, e, f, x, y, z;
        a = this->plane_normal[0];
        b = this->plane_normal[1];
        c = this->plane_normal[2];

        d = this->updated_plane_origin(0);
        e = this->updated_plane_origin(1);
        f = this->updated_plane_origin(2);

        x = slave_pnt(0);
        y = slave_pnt(1);
        z = slave_pnt(2);
        double t = (a*d - a*x + b*e - b*y + c*f - c*z)/(a*a + b*b + c*c);

        master_pnt(0) = x + t*a;
        master_pnt(1) = y + t*b;
        master_pnt(2) = z + t*c;

        double gap  = scalar_product((slave_pnt - master_pnt),this->plane_normal);  
        if (point_on_spatular(master_pnt))
        {
            if (gap<0)
            {
                std::cout << "Gap between spatular and slave point:  " 
                                    << slave_pnt
                                    << " is "
                                    << gap
                                    << std::endl; 
                std::cout << "Master point:  " 
                                    << master_pnt
                                    << std::endl;
            }
            return gap;
        }
        return 1e6;       
    }

    template <int dim>
    bool Spatular<dim>::
    point_in_square(const dealii::Tensor<1,dim> & rot_master_pnt)
    {
        Tensor<1,dim-1> vert_a, vert_b, vert_c, vert_d, two_d_pnt; 

        two_d_pnt[0] = rot_master_pnt[0];       
        two_d_pnt[1] = rot_master_pnt[1];

        vert_a[0] = this->updated_vertices[0][0];
        vert_a[1] = this->updated_vertices[0][1];
        vert_b[0] = this->updated_vertices[1][0];
        vert_b[1] = this->updated_vertices[1][1];
        vert_c[0] = this->updated_vertices[2][0];
        vert_c[1] = this->updated_vertices[2][1];
        vert_d[0] = this->updated_vertices[3][0];
        vert_d[1] = this->updated_vertices[3][1];

        double vert_amab = scalar_product(two_d_pnt-vert_a,vert_b-vert_a);
        double vert_abab = scalar_product(vert_b-vert_a,vert_b-vert_a);

        double vert_amad = scalar_product(two_d_pnt-vert_a,vert_d-vert_a);
        double vert_adad = scalar_product(vert_d-vert_a,vert_d-vert_a);

        if ( ( (0. < vert_amab) && (vert_amab < vert_abab))  && ( (0. < vert_amad) && (vert_amad < vert_adad) ) )
            { 
                return true;
            } 
        return false;
    }

    template <int dim>
    bool Spatular<dim>::
    point_on_spatular(const dealii::Point<dim> & master_pnt)
    {

        Tensor<1,dim> rot_master_pnt(master_pnt);
        rot_master_pnt = this->rotation_matrix*rot_master_pnt;

        // efilog(Verbosity::verbose) << "Master point :  " << master_pnt << std::endl;
        // efilog(Verbosity::verbose) << "Rotated and moved master point :  " << rot_master_pnt << std::endl;        

        return this->point_in_square(rot_master_pnt);
    }

    template <int dim>
    void Spatular<dim>::
    move_vertices(double disp)
    {
        efilog(Verbosity::verbose) << "Scapular vertices updated :  " << std::endl;
        for (int i = 0; i < this->vertices.size(); i++)
            {
                efilog(Verbosity::verbose) << this->vertices[i] << " --> " ;
                updated_vertices[i] = vertices[i] + disp*this->plane_normal;
                updated_vertices[i] = this->rotation_matrix*this->updated_vertices[i];
                efilog(Verbosity::verbose) << updated_vertices[i] << std::endl;
            }
    }

    template <int dim>
    void Spatular<dim>::
    set_plane_normal(dealii::Tensor<1, dim> normal)
    {
        // this muct be the outward pointing normal
        this->plane_normal = normal;
    }

    template <int dim>
    void Spatular<dim>::
    set_plane_origin(dealii::Point<dim> origin)
    {

        this->plane_origin = origin;
    }

    template <int dim>
    void Spatular<dim>::
    set_vertices(std::vector<dealii::Tensor<1, dim>> verts)
    {
        this->vertices = verts;
    }

    template <int dim>
    void Spatular<dim>::
    set_rotation_matrix()
    {
        Tensor<1,dim> normal = this->plane_normal;
        double nx = normal[0];
        double ny = normal[1];
        double nz = normal[2];
        double base = std::sqrt( (nx*nx) + (ny*ny) );

        this->rotation_matrix = 0.;

        if (base < 1e-6)
        {
            this->rotation_matrix[0][0] = 1.;
            this->rotation_matrix[1][1] = 1.;
            this->rotation_matrix[2][2] = 1.;
        } else {

            this->rotation_matrix[0][0] = ny/base;
            this->rotation_matrix[0][1] = -1.0*nx/base;
            this->rotation_matrix[0][2] = 0;
            this->rotation_matrix[1][0] = (nx*nz)/base;
            this->rotation_matrix[1][1] = (ny*nz)/base;
            this->rotation_matrix[1][2] = -1.0*base;
            this->rotation_matrix[2][0] = nx;
            this->rotation_matrix[2][1] = ny;
            this->rotation_matrix[2][2] = nz;

        }

        efilog(Verbosity::verbose) << "Rotation matrix calculated \n["
                                << " [" << this->rotation_matrix[0][0] << ", " << this->rotation_matrix[0][1] << ", " << this->rotation_matrix[0][2] << "] \n"
                                << " [" << this->rotation_matrix[1][0] << ", " << this->rotation_matrix[1][1] << ", " << this->rotation_matrix[1][2] << "] \n"
                                << " [" << this->rotation_matrix[2][0] << ", " << this->rotation_matrix[2][1] << ", " << this->rotation_matrix[2][2] << "] ]"
                                << std::endl;
    }

    template <int dim>
    void Spatular<dim>::
    update(const double displacement)
    {
        efilog(Verbosity::verbose) << "Spaular position moved by " 
                                << displacement 
                                << " units"
                                << std::endl;

        this->updated_plane_origin = this->plane_origin + displacement*this->plane_normal;
        efilog(Verbosity::verbose) << "Scapular moved to :  " << this->updated_plane_origin << std::endl;
        this->move_vertices(displacement);
    }

// Instantiation
template class Spatular<2>;
template class Spatular<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Spatular,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Spatular,3));
}