/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The efi library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 *  Author: Stefan Kaessmair
 */

#ifndef SRC_MYLIB_INCLUDE_EFI_BASE_AUTOMATIC_DIFFERENTIATION_H_
#define SRC_MYLIB_INCLUDE_EFI_BASE_AUTOMATIC_DIFFERENTIATION_H_

// deal.II headers
#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>
#include <deal.II/differentiation/ad/ad_number_traits.h>
#include <deal.II/differentiation/ad/adolc_number_types.h>
#include <deal.II/differentiation/ad/adolc_product_types.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>



namespace efi {

namespace AD
{
    using NumberTypes = dealii::Differentiation::AD::NumberTypes;

    template <class ScalarType, NumberTypes ADNumberTypeCode>
    using NumberTraits = dealii::Differentiation::AD::NumberTraits<ScalarType,ADNumberTypeCode>;

    template <class Number>
    using ADNumberTraits = dealii::Differentiation::AD::ADNumberTraits<Number>;

    template <class Number>
    using is_ad_number = dealii::Differentiation::AD::is_ad_number<Number>;

    template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
    using CellLevelBase = dealii::Differentiation::AD::CellLevelBase<ADNumberTypeCode,ScalarType>;

    template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
    using ResidualLinearization = dealii::Differentiation::AD::ResidualLinearization<ADNumberTypeCode,ScalarType>;

    template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
    using EnergyFunctional = dealii::Differentiation::AD::EnergyFunctional<ADNumberTypeCode,ScalarType>;



    namespace efi_internal {

    template <class Type>
    struct Cloneable
    {
        // Default constructor.
        Cloneable () = default;

        // Clone function.
        virtual
        Cloneable<Type>*
        clone () const = 0;

        // Virtual destructor.
        virtual
        ~Cloneable() = default;

        // Content member variable.
        std::unique_ptr<Type> content;
    };



    template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
    struct Cloneable<ResidualLinearization<ADNumberTypeCode,ScalarType>> :
        Cloneable<CellLevelBase<ADNumberTypeCode,ScalarType>>
    {
        // Default constructor.
        Cloneable () = default;

        // Clone function.
        Cloneable<ResidualLinearization<ADNumberTypeCode,ScalarType>>*
        clone () const override final;
    };



    template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
    struct Cloneable<EnergyFunctional<ADNumberTypeCode,ScalarType>> :
        Cloneable<CellLevelBase<ADNumberTypeCode,ScalarType>>
    {
        // Default constructor.
        Cloneable () = default;

        // Clone function.
        Cloneable<EnergyFunctional<ADNumberTypeCode,ScalarType>>*
        clone () const override final;
    };

    }//namespace efi_internal


    template <class Number>
    struct CellLevelBaseHolder
    {
        using held_type = efi_internal::Cloneable<CellLevelBase<ADNumberTraits<Number>::type_code,
                                              typename ADNumberTraits<Number>::scalar_type>>;

        // Default constructor.
        CellLevelBaseHolder () = default;

        // Move constructor.
        CellLevelBaseHolder (CellLevelBaseHolder<Number> &&) = default;

        // Copy constructor.
        CellLevelBaseHolder (const CellLevelBaseHolder<Number> &other);

        // Default destructor.
        ~CellLevelBaseHolder() = default;

        // Assignment operator.
        CellLevelBaseHolder<Number>&
        operator= (const CellLevelBaseHolder &rhs);

        // Return a reference to the held AD:.CellLevelBase object.
        CellLevelBase<ADNumberTraits<Number>::type_code, typename ADNumberTraits<Number>::scalar_type>&
        get ();

        // Return a const reference to the held AD:.CellLevelBase object.
        const CellLevelBase<ADNumberTraits<Number>::type_code, typename ADNumberTraits<Number>::scalar_type>&
        get () const;

        // Set a new object derived from AD::CellLevelBase.
        // It is initialized with the given arguments.
        template <class CellLevelDerivedType, class ... Args>
        void
        set (Args &&...args);

        std::unique_ptr<held_type> held;
    };

}// namespace AD



template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
AD::efi_internal::Cloneable<AD::EnergyFunctional<ADNumberTypeCode,ScalarType>>*
AD::efi_internal::Cloneable<AD::EnergyFunctional<ADNumberTypeCode,ScalarType>>::
clone () const
{
    auto cloned = new Cloneable<EnergyFunctional<ADNumberTypeCode,ScalarType>>;

    if (this->content)
    {
        cloned->content.reset(
                new EnergyFunctional<ADNumberTypeCode,ScalarType>
                    (this->content->n_independent_variables()));

        std::vector<ScalarType> local_dof_values;
        for (auto &ad_value : this->content->get_sensitive_dof_values())
            local_dof_values.push_back(NumberTraits<ScalarType,ADNumberTypeCode>::get_scalar_value(ad_value));

        cloned->content->register_dof_values(local_dof_values);

        return cloned;
    }
    else
        return cloned;
}



template <AD::NumberTypes ADNumberTypeCode, class ScalarType>
AD::efi_internal::Cloneable<AD::ResidualLinearization<ADNumberTypeCode,ScalarType>>*
AD::efi_internal::Cloneable<AD::ResidualLinearization<ADNumberTypeCode,ScalarType>>::
clone () const
{
    auto cloned = new Cloneable<ResidualLinearization<ADNumberTypeCode,ScalarType>>;

    if (this->content)
    {
        cloned->content.reset(
                new AD::ResidualLinearization<ADNumberTypeCode,ScalarType>
                    (this->content->n_independent_variables(), this->content->n_dependent_variables()));

        std::vector<ScalarType> local_dof_values;
        for (auto &ad_value : this->content->get_sensitive_dof_values())
            local_dof_values.push_back(NumberTraits<ScalarType,ADNumberTypeCode>::get_scalar_value(ad_value));

        cloned->content->register_dof_values(local_dof_values);

        return cloned;
    }
    else
        return cloned;
}



template <class Number>
AD::CellLevelBaseHolder<Number>::
CellLevelBaseHolder (const CellLevelBaseHolder<Number> &other)
{
    *this=other;
}



template <class Number>
AD::CellLevelBaseHolder<Number>&
AD::CellLevelBaseHolder<Number>::
operator= (const CellLevelBaseHolder<Number> &rhs)
{
    if (rhs.held)
        this->held.reset(rhs.held->clone());
    else
        this->held.reset();

    return *this;
}



template <class Number>
AD::CellLevelBase<AD::ADNumberTraits<Number>::type_code, typename AD::ADNumberTraits<Number>::scalar_type>&
AD::CellLevelBaseHolder<Number>::
get ()
{
    Assert (held,dealii::ExcNotInitialized());
    Assert (held->content,dealii::ExcNotInitialized());
    return *(held->content);
}



template <class Number>
const AD::CellLevelBase<AD::ADNumberTraits<Number>::type_code, typename AD::ADNumberTraits<Number>::scalar_type>&
AD::CellLevelBaseHolder<Number>::
get () const
{
    Assert (held,dealii::ExcNotInitialized());
    Assert (held->content,dealii::ExcNotInitialized());
    return *(held->content);
}



template <class Number>
template <class CellLevelDerivedType, class ... Args>
void
AD::CellLevelBaseHolder<Number>::
set (Args &&...args)
{
    if (!held)
        held.reset(new efi_internal::Cloneable<CellLevelDerivedType>);
    held->content.reset (new CellLevelDerivedType(std::forward<Args>(args)...));
}

}// namespace efi

#endif /* SRC_MYLIB_INCLUDE_EFI_BASE_AUTOMATIC_DIFFERENTIATION_H_ */
