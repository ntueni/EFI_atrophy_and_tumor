

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_SPATULAR_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_SPATULAR_H_

// deal.II headers
#include <deal.II/base/parameter_handler.h>

// efi headers
#include <efi/grid/obstacle.h>
#include <efi/factory/registry.h>

namespace efi {
    
    template<int dim>
    class Spatular : public Obstacle<dim>
    {
    public:
        EFI_REGISTER_AS_BASE;

        Spatular(const std::string &subsection_name,
              const std::string &unprocessed_input);

        ~Spatular() = default;

        /// Declare parameters to the given parameter handler.
        /// @param[out] prm The parameter handler for which we want to declare
        /// the parameters.
        void
        declare_parameters (dealii::ParameterHandler &prm) final;

        /// Parse the parameters stored int the given parameter handler.
        /// @param[in] prm The parameter handler whose parameters we want to
        /// parse.
        void
        parse_parameters (dealii::ParameterHandler &prm) final;

        void create();

        double find_master_pnt(const dealii::Point<dim> & slave_pnt, 
            dealii::Point<dim> & master_pnt, bool);

        void update(const double);
    private:


        dealii::Tensor<1, dim> plane_normal;
        dealii::Point<dim> plane_origin;
        dealii::Point<dim> updated_plane_origin;
        dealii::Tensor<2, dim> rotation_matrix;

        std::vector<dealii::Tensor<1, dim>> vertices;
        std::vector<dealii::Tensor<1, dim>> updated_vertices;
    
        void set_plane_normal(dealii::Tensor<1, dim>);
        void set_plane_origin(dealii::Point<dim>);
        void set_vertices(std::vector<dealii::Tensor<1, dim>>);
        void move_vertices(double disp);

        void set_rotation_matrix();

        bool point_in_square(const dealii::Tensor<1,dim> &);
        bool point_on_spatular(const dealii::Point<dim> &);
    };


    //////////////////////////// SPATULAR inline fxs ///////////////////////////


    template <int dim>
    void
    Spatular<dim>::
    declare_parameters (dealii::ParameterHandler &prm)
    {   

        using namespace dealii;

        Obstacle<dim>::declare_parameters(prm);

        using T = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;        
        Tensor<1,dim> tensor;
        
        prm.declare_entry("normal",T::to_string(tensor),*T::to_pattern(),"Documentation");

        using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;        
        Point<dim> point;

        prm.declare_entry("origin",P::to_string(point),*P::to_pattern(),"Documentation");

        prm.declare_entry("vertex1",T::to_string(point),*T::to_pattern(),"Documentation");
        prm.declare_entry("vertex2",T::to_string(point),*T::to_pattern(),"Documentation");
        prm.declare_entry("vertex3",T::to_string(point),*T::to_pattern(),"Documentation");
        prm.declare_entry("vertex4",T::to_string(point),*T::to_pattern(),"Documentation");

        efilog(Verbosity::verbose) << "Spaular obstacle finished "
                                    "declaring parameters."
                                << std::endl;
    }

    template <int dim>
    void
    Spatular<dim>::
    parse_parameters (dealii::ParameterHandler &prm)
    {   
        using namespace dealii;
        Obstacle<dim>::parse_parameters(prm);

        using T = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;

        dealii::Tensor<1, dim> normal;
        normal = T::to_value(prm.get ("normal"));
        this->set_plane_normal(normal);

        using P = dealii::Patterns::Tools::Convert<Tensor<1,dim>>;
        dealii::Point<dim> origin;

        origin = P::to_value(prm.get("origin"));
        this->set_plane_origin(origin);

        std::vector<Tensor<1, dim>> vertices(4);
        vertices[0] = T::to_value(prm.get("vertex1"));
        vertices[1] = T::to_value(prm.get("vertex2"));
        vertices[2] = T::to_value(prm.get("vertex3"));
        vertices[3] = T::to_value(prm.get("vertex4"));
        this->set_vertices(vertices);

        this->set_rotation_matrix();
        this->move_vertices(0.0);

        efilog(Verbosity::verbose) << "Spatular obstacle finished "
                                  "parsing parameters."
                               << std::endl;
    }
}
#endif