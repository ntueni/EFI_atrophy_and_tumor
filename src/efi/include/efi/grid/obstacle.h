

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_OBSTACLE_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_OBSTACLE_H_

#include <deal.II/base/point.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/patterns.h>

#include <efi/factory/registry.h>


// Base class for any obstacle
// methods that must be implemented:
// - declare_parameters
// - parse_parameters
// - create
// - find_master_point 
namespace efi {

    template<int dim>
    class Obstacle : public dealii::ParameterAcceptor 
    {
    public:
        EFI_REGISTER_AS_BASE;

        // static int cellCount;

        Obstacle(const std::string &subsection_name,
                      const std::string &unprocessed_input);

        ~Obstacle() = default;

        void set_contact_boundary(dealii::types::boundary_id);
        
        int get_contact_boundary_id() const;

        void set_penalty_parameter(const double);

        double get_penalty_parameter() const;

        virtual
        void parse_parameters (dealii::ParameterHandler &prm);

        virtual
        void declare_parameters (dealii::ParameterHandler &prm);

        virtual
        void create();

        virtual
        double find_master_pnt(const dealii::Point<dim> & slave_pnt, 
            dealii::Point<dim> & master_pnt, bool);

        virtual
        void update(const double);

    protected:
        double c;
        double gap;
        dealii::types::boundary_id contact_boundary_id;
        std::string section_path_str;        
    };

    //---------------------- INLINE AND TEMPLATE FUNCTIONS -----------------------//

    template <int dim>
    inline
    Obstacle<dim>::
    Obstacle(const std::string &subsection_name,
                  const std::string &)
    :
    dealii::ParameterAcceptor (subsection_name),
    section_path_str (get_section_path_str(this->get_section_path()))
    {
        // Do Nothing
    }

    template <int dim>
    void
    Obstacle<dim>::
    declare_parameters (dealii::ParameterHandler &prm)
    {
        prm.declare_entry("penalty","0",dealii::Patterns::Double());
    }

    template <int dim>
    void
    Obstacle<dim>::
    parse_parameters (dealii::ParameterHandler &prm)
    {
        auto penalty_parameter = prm.get_double ("penalty");
        this->set_penalty_parameter(penalty_parameter);
    }

    template <int dim>
    inline
    void Obstacle<dim>::
    set_penalty_parameter(const double penalty)
    {
        this->c = penalty;
    }


    template <int dim>
    inline
    double Obstacle<dim>::
    get_penalty_parameter() const
    {
        return this->c;
    }


    template <int dim>
    inline
    void Obstacle<dim>::
    set_contact_boundary(dealii::types::boundary_id boundary_id)
    {
        this->contact_boundary_id = boundary_id;
    }

    template <int dim>
    inline
    int Obstacle<dim>::
    get_contact_boundary_id() const
    {
        return this->contact_boundary_id;
    }

    template <int dim>
    inline
    void Obstacle<dim>::
    update(const double)
    {
        // Do nothing
    }

    template <int dim>
    inline
    void Obstacle<dim>::
    create() 
    {
        efilog(Verbosity::verbose) << "Contact will be applied"
                                << std::endl;
    }

    template <int dim>
    inline
    double Obstacle<dim>::
    find_master_pnt(const dealii::Point<dim> &, 
        dealii::Point<dim> & , bool) 
    {
        // Do nothing
        return 1e10;
    }


}    
#endif