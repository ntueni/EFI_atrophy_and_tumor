#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_OBSTACLE_FACTORY_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_OBSTACLE_FACTORY_H_

// efi header
#include <efi/base/utility.h>
#include <efi/base/factory_tools.h>
#include <efi/grid/obstacle.h>
#include <efi/grid/skull.h>
#include <efi/grid/spatular.h>


namespace efi
{
    // Factory pattern implementation for
    // Geometries.
    template <int dim>
    class ObstacleFactory
    {
    public:

        // Create a new Geometry instance of
        // specified by the "type" stored in the
        // Sepcifications.
        static
        Obstacle<dim>*
        create (const std::vector<std::string>     &section_path,
                const FactoryTools::Specifications &specs,
                const std::string                  &unprocessed_input);

        // Create a new Geometry instance of
        // specified by the string type_str.
        // The arguments args... are forwarded to the
        // contitutive model constructor.
        // An error of type efi::ExcNotConstructible
        // is thrown if the given string does not match
        // any option.
        template <class ... Args>
        static
        Obstacle<dim>*
        create (const std::string& type_str,
                Args &&... args);

        // return the keyword which
        static
        std::string
        keyword ();

        // Return a string that specifies the list of
        // of allowed options. The options are separated
        // by '|' no addional spaces are added (e.g.
        // "model1|model2|model3").
        static
        std::string
        get_names ();
    };



    ///////////////////////////////////////////////////////////////////////////////
    // IMPLEMENTATION
    ///////////////////////////////////////////////////////////////////////////////



    template <int dim>
    Obstacle<dim>*
    ObstacleFactory<dim>::
    create (const std::vector<std::string>     &section_path,
            const FactoryTools::Specifications &specs,
            const std::string                  &unprocessed_input)
    {
        efilog(Verbosity::verbose) << "Entered Obstacle factory"
                               << std::endl;
        return ObstacleFactory<dim>::create (
                specs.get("type"),
                get_section_path_str(section_path) + "/" +
                    FactoryTools::get_subsection_name (keyword(),specs),
                unprocessed_input);
    }



    template <int dim>
    template <class ... Args>
    Obstacle<dim>*
    ObstacleFactory<dim>::
    create (const std::string& type_str,
            Args &&... args)
    {
        if (type_str == "skull")
        {
            using obstacle_type = Skull<dim>;

            return make_new_if_constructible<obstacle_type>(std::forward<Args>(args)...);
        }
        else if (type_str == "spatular")
        {
            using obstacle_type = Spatular<dim>;

            return make_new_if_constructible<obstacle_type>(std::forward<Args>(args)...);
        }
        else
            AssertThrow (false, ExcNotConstructible ());
    }

    template <int dim>
    inline
    std::string
    ObstacleFactory<dim>::
    keyword ()
    {
        return "obstacle";
    }



    template <int dim>
    inline
    std::string
    ObstacleFactory<dim>::
    get_names ()
    {
        return "skull|spatular";
    }

}//namespace efi


#endif /* SRC_MYLIB_INCLUDE_EFI_GRID_OBSTACLE_FACTORY_H_ */