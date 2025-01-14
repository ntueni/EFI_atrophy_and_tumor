
#include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/tria.h>
// #include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/base/exceptions.h>


#include <efi/grid/obstacle.h>
#include <efi/grid/skull.h>
#include <efi/factory/registry.h>
#include <efi/lab/sample.h>

#include <fstream>
#include <iostream>

namespace efi {

    using namespace dealii;

    // template <int dim>
    // int Obstacle<dim>::cellCount = 0;

    template <int dim>
    Skull<dim>::
    Skull(const std::string &subsection_name,
              const std::string &unprocessed_input)
    :
        Obstacle<dim>(subsection_name,unprocessed_input)
    {
        efilog(Verbosity::verbose) << "skull Created ( "
                               << subsection_name
                               << ")."<< std::endl;
        std::vector<double> displacements = {5,5,5};
        this->delta = displacements;
        this->searchTree = new BST<dim>(0);
    }

    template <int dim>
    void Skull<dim>::
    create()
    {   
        Obstacle<dim>::create();

        this->gap = 0.0;

        efi::efilog(Verbosity::normal) << "Importing Skull from "<< this->boundary_file << std::endl;
        std::ifstream istream(this->boundary_file);
        dealii::GridIn<spacedim,dim> gridIn;
        gridIn.attach_triangulation(this->testTriangulation);
        gridIn.read_ucd(istream);

        // this->print_surface("/calculate/efiSim1F/build/out/Boundary.vtu");

        efi::efilog(Verbosity::normal) << "Boundary grid IMPORTED" << std::endl;

        this->create_capture_boxes();
        std::vector<CollectionBoxes<dim>> sortedboxes;
        this->sort_capture_boxes(0, sortedboxes);  
        const int comp = 0;
        this->boxesToBST(sortedboxes, comp, *this->searchTree);
        efi::efilog(Verbosity::verbose) << "Search tree for faces CREATED" << std::endl; 
    }

    template <int dim>
    double Skull<dim>::
    find_master_pnt(const Point<dim> & slave_pnt, 
        Point<dim> & master_pnt, bool print)
    {
        std::vector<CaptureBox<dim>> boxes;
        this->search_tree(slave_pnt, boxes);
        if (print)
            efilog(Verbosity::debug) << " number of boxes found: " << boxes.size() << std::endl;
        Assert(boxes.size(), dealii::ExcZero());

        std::vector<dealii::CellId> closeCellIds;
        this->find_absolute_closest(slave_pnt, boxes, closeCellIds);

        Assert(closeCellIds.size(), dealii::ExcZero());

        if (print)
            efilog(Verbosity::debug) << closeCellIds[0] << " was found to be the absolute closest face" << std::endl;
        
        return this->calculate_min_gap(master_pnt, slave_pnt, closeCellIds, print);        
    }

    template <int dim>
    void Skull<dim>::
    set_delta(const std::vector<double> & displacements)
    {
        this->delta = displacements;
        this->delta_set = true;
    }


    template <int dim>
    void Skull<dim>::
    set_boundary_file(const std::string & filename)
    {
        this->boundary_file = filename;
    }

    template <int dim>
    void Skull<dim>::
    print_surface(std::string fileName) const
    {    
        std::ofstream out(fileName);
        GridOut       grid_out;
        grid_out.write_vtu(this->testTriangulation, out);
    }

    template <int dim>
    void Skull<dim>::
    create_capture_boxes()
    {
        for( const auto & cell: this->testTriangulation.active_cell_iterators())
        {
            // efilog(Verbosity::debug) << "face at boundary" << std::endl;
            Point<dim> lower;
            Point<dim> upper;
            BoundingBox<dim> box = cell->bounding_box();
            for (unsigned int i = 0; i < dim; i++)
            {
                lower(i) = box.lower_bound(i) - this->delta[i];
                upper(i) = box.upper_bound(i) + this->delta[i];
            }
            std::pair<Point<dim>, Point<dim>> limits;
            limits.first = lower;
            limits.second = upper;
            std::stringstream ss;
            dealii::CellId cell_id = cell->id();
            // efilog(Verbosity::debug) << "Capture boxes id: " <<  cell_id.to_string() << std::endl;
        
            // std::vector<Point<dim>> vertices(4);
            // for (unsigned int v = 0; i < 4; v++)
            // {
            //    vertices[v] = cell->vertex(v) 
            // }
            CaptureBox<dim> cp(BoundingBox<dim>(limits), cell_id);
            capture_boxes.push_back(cp);                                                                    
        };
        efi::efilog(Verbosity::verbose) << capture_boxes.size() << " Capture boxes created" << std::endl;
    }

    template <int dim>
    void Skull<dim>::
    sort_capture_boxes(int comp,
                       std::vector<CollectionBoxes<dim>> & sortedBoxes)
    {
         std::map<signed int, CollectionBoxes<dim>> collectionBoxMap;
        for (CaptureBox<dim> b : this->capture_boxes)
        {
            const signed int key = std::rint(b.upperLimit(comp) * 1000);
            CollectionBoxes<dim> boxCollection(comp);
            if (collectionBoxMap.count(key))
                boxCollection = collectionBoxMap.find(key)->second;

            boxCollection.addBox(b);
            collectionBoxMap.erase(key);
            collectionBoxMap.insert(std::pair<signed int, CollectionBoxes<dim>>(key, boxCollection));
        }

        // efilog(Verbosity::debug) << "Keys in map: \n";
        // for (auto itr = collectionBoxMap.begin(); itr != collectionBoxMap.end(); itr++ )
        //   efilog(Verbosity::debug) << itr->first << ", with " << itr->second.size() << " entries\n";
        // efilog(Verbosity::debug) << std::endl;

        for (auto itr = collectionBoxMap.begin(); itr != collectionBoxMap.end(); itr++)
            sortedBoxes.push_back(itr->second);       
    }

    template <int dim>
    void Skull<dim>::
    boxesToBST(std::vector<CollectionBoxes<dim>> &sortedBoxes, const int comp,
                                   BST<dim> &bst)
    {
        const int sortedBoxSize = sortedBoxes.size();
        int median = std::ceil(sortedBoxSize / 2);
        std::vector<CollectionBoxes<dim>> node_middle;
        node_middle.push_back(sortedBoxes.at(median));
        std::vector<CollectionBoxes<dim>> node_left(sortedBoxes.begin(), sortedBoxes.begin() + median);
        auto mid_it = sortedBoxes.begin() + median + 1;
        std::vector<CollectionBoxes<dim>> node_right(mid_it, sortedBoxes.end());

        double cuttingLine = node_middle.at(0).getMaxVal();
        int nodes_to_remove_left = 0;
        int nodes_to_remove_right = 0;
        for (auto it = node_left.rbegin(); it != node_left.rend(); it++)
        {
            double leftUpperBound = it->getMaxVal();
            if ((leftUpperBound - cuttingLine) > -1e-6)
            {
                node_middle.push_back(*it);
                nodes_to_remove_left++;
            }
            else
                break;
        }
        if (nodes_to_remove_left > 0)
            node_left.erase(node_left.end() - nodes_to_remove_left, node_left.end());

        for (auto it = node_right.begin(); it != node_right.end(); it++)
        {
            double rightLowerBound = it->getMinVal();
            if ((rightLowerBound - cuttingLine) < 1e-6)
            {
                node_middle.push_back(*it);
                nodes_to_remove_right++;
            }
            else
                break;
        }
        if (nodes_to_remove_right > 0)
            node_right.erase(node_right.begin(), node_right.begin() + nodes_to_remove_right);

        Node<dim> *newNode = new Node<dim>(cuttingLine);

        int newComp = comp + 1;
        if (newComp < dim)
        {
            for (auto iter = node_middle.begin(); iter != node_middle.end(); iter++)
            {
                this->capture_boxes = iter->getCaptureBoxes();
                std::vector<CollectionBoxes<dim>> newSortedboxes;
                this->sort_capture_boxes(newComp, newSortedboxes);
                BST<dim> *searchTree = new BST<dim>(newComp);
                boxesToBST(newSortedboxes, newComp, *searchTree);
                iter->setNextCoordBST(searchTree);
            }
        }
        newNode->addBoxes(node_middle);
        bst.insert(*newNode);

        if (node_left.size() > 0)
        {
            this->boxesToBST(node_left, comp, bst);
        }
        if (node_right.size() > 0)
        {
            this->boxesToBST(node_right, comp, bst);
        }
    }

    template <int dim>
    void Skull<dim>::
    search_tree(const Point<dim> & query_point, std::vector<CaptureBox<dim>> & boxes)
    {
        int *component = new int(0);
        this->searchTree->search(query_point, component, boxes);
    }

    template <int dim>
    void Skull<dim>::
    find_absolute_closest(const Point<dim> & qp, 
    std::vector<CaptureBox<dim>> & close_boxes,
    std::vector<dealii::CellId> & close_cells)
    {
        // Change to find closest node within thte selected close boxes
        // efilog(Verbosity::debug)  << "Querying point: " << qp << std::endl;
        std::map<double, dealii::CellId> distance_to_cell;
        dealii::CellId minFaceId;
        for (const auto box : close_boxes)
        {
            const dealii::CellId id = box.getCellId();
            // efilog(Verbosity::debug) << " Cell_Id: " << id.to_string() << std::endl;
            const auto cell = this->testTriangulation.create_cell_iterator(id);

            auto centroid = cell->center();
            double dist = qp.distance(centroid);
            distance_to_cell.insert(std::make_pair(dist,id));
        }
        std::vector<double> keys, value;
        for(auto & it : distance_to_cell)
            keys.push_back(it.first);

        sort(keys.begin(), keys.end());

        for (auto key: keys)
            close_cells.push_back(distance_to_cell.find(key)->second);
    }

    template<int dim>
    double Skull<dim>::
    calculate_min_gap(Point<dim> & master_pnt, 
        const Point<dim> & slave_pnt, 
        const std::vector<dealii::CellId> & closeFaces,
        bool print)
    {
        int cellNo = 0;
        dealii::CellId cellId = closeFaces[cellNo];
        if (print)
            efilog(Verbosity::debug) << "Finding min gap between point and face: " << cellId.to_string() <<std::endl;
        bool converged = false;
        double deltazi = 0.;
        bool newZi = false;
        dealii::Point<dim>& Nx = master_pnt;
        std::vector<std::vector<double>> shape_fx_nodes{{-1, -1},
                                                            {1, -1},
                                                            {1, 1},
                                                            {-1, 1}};
        double tol = 1e-10;
        while(!converged)
        {
            double gap = 0.;
            const auto cell_face = this->testTriangulation.create_cell_iterator(cellId);
            // if (cell_face->material_id() == 4)
            //     {
            //         cellCount++;
            //         return false;
            //     }
            std::vector<Point<dim>> nodes_coords(4);
            for (const auto v : cell_face->vertex_indices())
            {
                nodes_coords[v] = cell_face->vertex(v);
                if (slave_pnt.distance(nodes_coords[v]) < 1e-9)
                {
                    master_pnt = nodes_coords[v];
                    return 0.0;
                }
            }

            Point<dim> temp = nodes_coords[3];
            nodes_coords[3] = nodes_coords[2];
            nodes_coords[2] = temp;

            Vector<double> initial_guess(2);
            initial_guess(0) = 0. + deltazi*0.3;
            initial_guess(1) = 0.1 + deltazi;
            Vector<double> delta_guess(2);
            Vector<double> dNddAlpha(3);
            dNddAlpha = 0.0;
            std::vector<Tensor<1, dim>> dNdAlpha(2);
            Nx.clear();

            Vector<double> rhs(2);
            for (unsigned int n = 0; n < nodes_coords.size(); n++)
                for (int i = 0; i < dim; i++)
                    dNddAlpha(i) += 0.25 * shape_fx_nodes[n][0] * shape_fx_nodes[n][1] * nodes_coords[n](i);


            FullMatrix<double> dNdd(2,6);
            dNdd = 0.;
            dNdd(0,3) = dNddAlpha(0);
            dNdd(0,4) = dNddAlpha(1);
            dNdd(0,5) = dNddAlpha(2);
            dNdd(1,0) = dNddAlpha(0);
            dNdd(1,1) = dNddAlpha(1);
            dNdd(1,2) = dNddAlpha(2);

            Tensor<2,dim-1> tangent_matrix;
            converged = false;
            int iter = 0;
            while (!converged && iter < 20)
            {
                if (print)
                    efilog(Verbosity::debug)  << "iter: " << iter << std::endl;
                delta_guess = 0.;
                dNdAlpha[0] = 0.;
                dNdAlpha[1] = 0.;
                Nx.clear();
                for (unsigned int n = 0; n < nodes_coords.size(); n++)
                    for (int i = 0; i < dim; i++)
                    {
                        Nx(i) += 0.25 * (1 + shape_fx_nodes[n][0] * initial_guess(0)) * (1 + shape_fx_nodes[n][1] * initial_guess(1)) * nodes_coords[n](i);
                        dNdAlpha[0][i] += 0.25 * shape_fx_nodes[n][0] * (1 + shape_fx_nodes[n][1] * initial_guess(1)) * nodes_coords[n](i);
                        dNdAlpha[1][i] += 0.25 * shape_fx_nodes[n][1] * (1 + shape_fx_nodes[n][0] * initial_guess(0)) * nodes_coords[n](i);
                    }

                tangent_matrix = 0;
                rhs = 0;

                for (int d = 0; d < dim; d++)
                    for (int k = 0; k < 2; k++)
                    {
                        for (int j = 0; j < 2; j++)
                        {
                            int nodeNo = 3*j+d;
                            tangent_matrix[k][j] += (dNdAlpha[k][d] * dNdAlpha[j][d]) - dNdd(k,nodeNo) * (slave_pnt[d] - Nx(d));
                        }
                        rhs(k) += -1. * dNdAlpha[k][d] * (slave_pnt[d] - Nx(d));
                    }
                Tensor<2,dim-1> tangent_inv;
                tangent_inv = invert(tangent_matrix);
                tangent_inv *= -1;
                delta_guess(0) = (tangent_inv[0][0]*rhs(0)) + (tangent_inv[0][1]*rhs(1));
                delta_guess(1) = (tangent_inv[1][0]*rhs(0)) + (tangent_inv[1][1]*rhs(1));
                initial_guess(0) += delta_guess(0);
                initial_guess(1) += delta_guess(1);
                iter++;
                // efilog(Verbosity::debug) << initial_guess << std::endl;
                // efilog(Verbosity::debug) << "rhs norm: " << rhs.l2_norm() << std::endl;
                // efilog(Verbosity::debug) << "initial_guess(1): " << initial_guess(1) << std::endl;
                if (rhs.l2_norm() < tol)
                    converged = true;
            }

            if (converged)
            {
                Tensor<1, dim> normal = cross_product_3d(dNdAlpha[0], dNdAlpha[1]);
                normal = normal / normal.norm();
                gap = scalar_product(normal, (slave_pnt - Nx));
                
                if (std::fabs(gap)>1.5)
                {
                    converged = false;
                    if (!newZi)
                    {
                        if (print)
                            efilog(Verbosity::debug) << "Poor convergence at first zi on cell " << cellId.to_string() << std::endl;
                        deltazi = 0.7;
                        newZi = true;
                        cellNo++;
                    } else
                    {     
                        if (print) 
                            efilog(Verbosity::debug) << "Still poor no convergence at a different zi for cell " << cellId.to_string() << std::endl;
                        deltazi = 0.0; 
                        newZi = false;            
                        cellId = closeFaces[cellNo];
                    }
                    if (print) 
                        {
                            efilog(Verbosity::debug) << "iterations: " << iter << std::endl;
                            efilog(Verbosity::debug) << "cellNo: " << cellId.to_string() << std::endl;
                            efilog(Verbosity::debug) << "gap: " << gap << std::endl;
                            efilog(Verbosity::debug) << "slave_pnt = [ " << slave_pnt << "];" << std::endl;
                            efilog(Verbosity::debug) << "node_coords = [ " << nodes_coords[0] << ";\n\t";
                            efilog(Verbosity::debug) << nodes_coords[1] << ";\n\t";
                            efilog(Verbosity::debug) << nodes_coords[2] << ";\n\t";
                            efilog(Verbosity::debug) << nodes_coords[3] << "];" << std::endl;
                        }
                } else if (cellNo>0)
                {
                    if (print) 
                    {
                        efilog(Verbosity::debug) << "Convergence for next cell" << std::endl;
                        efilog(Verbosity::debug) << "iterations: " << iter << std::endl;
                        efilog(Verbosity::debug) << "cellNo: " << cellId.to_string() << std::endl;
                        efilog(Verbosity::debug) << "gap: " << gap << std::endl;
                        efilog(Verbosity::debug) << "slave_pnt = [ " << slave_pnt << "];" << std::endl;
                        efilog(Verbosity::debug) << "node_coords = [ " << nodes_coords[0] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[1] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[2] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[3] << "];" << std::endl;
                    }
                    deltazi = 0.;
                    newZi = false;
                    return gap;
                } else {
                    return gap;
                }
            }
            else
            {
                if (print) 
                    efilog(Verbosity::debug) << "No convergence achieved for cell " << cellId.to_string();
                if (cellNo < (closeFaces.size()-1) || newZi)
                {
                    if (print) 
                    {
                        efilog(Verbosity::debug) << " at new zi value" << std::endl;
                        efilog(Verbosity::debug) << "slave_pnt = [ " << slave_pnt << "];" << std::endl;
                        efilog(Verbosity::debug) << "node_coords = [ " << nodes_coords[0] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[1] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[2] << ";\n\t";
                        efilog(Verbosity::debug) << nodes_coords[3] << "];" << std::endl;
                    }
                    deltazi = 0.;
                    newZi = false;
                    cellNo++;                    
                    cellId = closeFaces[cellNo];
                }
                else if (!newZi)
                {
                    if (print) 
                        efilog(Verbosity::debug) << " at original zi value, trying new zi value" << std::endl;
                    deltazi = 0.7;
                    newZi = true;
                } else 
                {
                    efilog(Verbosity::debug) <<"PROBLEM: NO convergence for any senario" << std::endl;
                    return 1.0e6;
                }
            }
        }
        return 1.0e6;
    }

    // template <int dim>
    // void Skull<dim>::create_dof_map()
    // {
    //     // efilog(Verbosity::debug) << "Entered create dof map" << std::endl;
    //     // this->dof_handler = &sample.get_dof_handler();
    //     // const auto &sample_fe = this->dof_handler->get_fe();

    //     // std::vector<bool> touched_dofs(this->dof_handler->n_dofs(),false);
    //     // Quadrature<dim-1> face_quadrature(sample_fe.get_unit_face_support_points());
    //     // FEFaceValues<dim> fe_values_face(sample_fe, face_quadrature, update_quadrature_points);

    //     // const unsigned int dofs_per_face = sample_fe.n_dofs_per_face();
    //     // const unsigned int n_face_q_points = face_quadrature.size();

    //     // std::vector<types::global_dof_index> dof_indices(dofs_per_face);

    //     for (const auto & cell : this->testTriangulation.vertex_iterator())
    //     {
    //         fe_values_face.reinit(cell);
    //         face->get_dof_indices(dof_indices);
    //         for (unsigned int q_point = 0; q_point<n_face_q_points; q_point += dim)
    //         {
    //             const int index = dof_indices[q_point];
    //             if (touched_dofs[index] == false)
    //             {
    //                 touched_dofs[index] = true;
    //                 Point<dim> slave_pnt = fe_values_face.quadrature_point(q_point);
    //                 std::vector<dealii::TriaIterator<dealii::DoFAccessor<2, dealii::DoFHandler<3, 3>, false> > > close_faces;
    //                 this->find_faces(slave_pnt,close_faces);
    //                 if (index == 0)
    //                 {
    //                     efilog(Verbosity::debug) << "index: " << index << std::endl;
    //                     efilog(Verbosity::debug) << "slave point: " << slave_pnt << std::endl;
    //                     }
    //                 this->dof_to_closest_faces_map.insert(std::make_pair(index,close_faces));
    //             }
    //         }
    //     }
        
    //     // efilog(Verbosity::debug) << "Keys in map: \n";
    //     // for (auto itr = this->dof_to_closest_faces_map.begin(); itr != this->dof_to_closest_faces_map.end(); itr++ )
    //     //   if (itr->first == 0)
    //     //     efilog(Verbosity::debug) << itr->first << ", with " << itr->second.size() << " cells\n";

    // }

// Instantiation
template class Skull<2>;
template class Skull<3>;

// Registration
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Skull,2));
EFI_REGISTER_OBJECT(EFI_TEMPLATE_CLASS(Skull,3));
}

