/*
 * Copyright (C) 2019 - 2020 by the emerging fields initiative 'Novel Biopolymer
 * Hydrogels for Understanding Complex Soft Tissue Biomechanics' of the FAU
 *
 * This file is part of the EFI library.
 *
 * The EFI library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * Author: Stefan Kaessmair
 */

#ifndef SRC_EFI_INCLUDE_EFI_WORKER_GENERAL_CELL_DATA_STORAGE_H_
#define SRC_EFI_INCLUDE_EFI_WORKER_GENERAL_CELL_DATA_STORAGE_H_

// stl headers
#include <string>
#include <map>

// deal.II headers
#include <deal.II/algorithms/general_data_storage.h>
#include <deal.II/grid/cell_id.h>

// boost headers
#include <boost/signals2.hpp>

namespace efi{


class
GeneralCellDataStorage
{
public:

    /// Default constructor.
    GeneralCellDataStorage () = default;

    /// Copy constructor.
    /// @param[in] storage GeneralCellDataStorage object to be copied.
    GeneralCellDataStorage (const GeneralCellDataStorage &storage) = default;

    /// Return a reference to the GeneralDataStorage of the current cell.
    template <class CellIteratorType>
    dealii::GeneralDataStorage&
    get_data (const CellIteratorType &cell);

    /// Return a const reference to the GeneralDataStorage of the current cell.
    template <class CellIteratorType>
    const dealii::GeneralDataStorage&
    get_data (const CellIteratorType &cell) const;

    /// Initialize the cell data storage with a given cell-range.
    template <class CellIteratorRange>
    void
    initialize (const CellIteratorRange &cells);

private:

    /// Map of cell-IDs to the stored data.
    std::map<dealii::CellId,dealii::GeneralDataStorage> cell_data;
};



//------------------- INLINE AND TEMPLATE FUNCTIONS -------------------



template <class CellIteratorType>
dealii::GeneralDataStorage&
GeneralCellDataStorage::
get_data (const CellIteratorType &cell)
{
    auto it = cell_data.find(cell->id());
    Assert (it != this->cell_data.end(), dealii::ExcMessage("Cell not found."));
    return it->second;
}



template <class CellIteratorType>
const dealii::GeneralDataStorage&
GeneralCellDataStorage::
get_data (const CellIteratorType &cell) const
{
    auto it = cell_data.find(cell->id());
    Assert (it != this->cell_data.end(), dealii::ExcMessage("Cell not found."));
    return it->second;
}



template <class CellIteratorRange>
void
GeneralCellDataStorage::
initialize (const CellIteratorRange &cells)
{
    cell_data.clear();
    for (const auto &cell : cells)
    {
        cell_data.emplace (cell->id(),dealii::GeneralDataStorage());
    }
}

}//namespace efi

#endif /* SRC_EFI_INCLUDE_EFI_WORKER_GENERAL_CELL_DATA_STORAGE_H_ */
