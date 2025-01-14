

#ifndef SRC_MYLIB_INCLUDE_EFI_GRID_SKULL_H_
#define SRC_MYLIB_INCLUDE_EFI_GRID_SKULL_H_

#include <deal.II/base/bounding_box.h>

#include <efi/base/global_parameters.h>
#include <efi/grid/obstacle.h>
#include <efi/factory/registry.h>

namespace efi {

    // Prototype for the sample class
    template <int dim>
    class Sample;

    template<int dim>
    class CaptureBox
    {
        public: 
        CaptureBox(dealii::BoundingBox<dim> box,  dealii::CellId);
        double upperLimit(const int) const;
        double lowerLimit(int) const;
        dealii::CellId getCellId() const;

        private:      
        dealii::BoundingBox<dim> boxDimensions;
        dealii::CellId cell_id;
    };

    template<int dim>
    class BST;

    template<int dim>
    class CollectionBoxes
    {
        public: 
        CollectionBoxes(int);
        void addBox(CaptureBox<dim> &);
        int size() const;
        double getMaxVal() const;
        double getMinVal() const;
        std::vector<CaptureBox<dim>> & getCaptureBoxes();
        BST<dim>* getNextCoordBST() const;
        void setNextCoordBST(BST<dim>*);

        private:
        std::vector<CaptureBox<dim>> captureBoxes;
        int componentOrderedOn;
        double maxVal;
        double minVal;
        BST<dim>* nextCoordBST = nullptr;
    };

    template<int dim>
    class Node
    {
        public:
            Node(double);
            void addBox(CollectionBoxes<dim> &);
            void addBoxes(std::vector<CollectionBoxes<dim>> & );
            std::vector<CollectionBoxes<dim>> getCollectionBoxes();
            double getCuttingLine();
            Node<dim>* getLeft() const;
            Node<dim>* getRight() const;
            void setLeft(Node<dim>*);
            void setRight(Node<dim>*);
        private:
            double cuttingLine;
            std::vector<CollectionBoxes<dim>> collectionBoxes;
            Node<dim>* left = nullptr;
            Node<dim>* right = nullptr;

    };

    template<int dim>
    class BST
    {
        public:
            BST(int);
            void search(const dealii::Point<dim>&, int*, std::vector<CaptureBox<dim>> & );
            void insert(Node<dim> &);
            Node<dim>* getRoot();
        private:
            int sortedComp;
            Node<dim>* root;
    };

    template<int dim>
    class Skull : public Obstacle<dim>
    {
    public:
        EFI_REGISTER_AS_BASE;
        
        Skull(const std::string &subsection_name,
              const std::string &unprocessed_input);        
        
        ~Skull() = default;

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

        void print_surface(std::string fileName) const;

        void set_delta(const std::vector<double>&);

        void create();

        double find_master_pnt(const dealii::Point<dim> & slave_pnt, 
            dealii::Point<dim> & master_pnt, bool);

    private:

        std::string boundary_file;

        static unsigned int uncovergedPoints;

        static constexpr unsigned int spacedim = dim - 1;
        dealii::Triangulation<spacedim, dim> testTriangulation;

        std::vector<double> delta;
        bool delta_set = false;
        std::vector<CaptureBox<dim>> capture_boxes;
        BST<dim> *searchTree = nullptr;

        void set_boundary_file(const std::string &);

        void create_capture_boxes();

        void sort_capture_boxes(int component,
                                std::vector<CollectionBoxes<dim>> & sortedBoxes);

        void boxesToBST(std::vector<CollectionBoxes<dim>> &sortedBoxes, const int comp, 
                        BST<dim> &searchTree);

        void search_tree(const dealii::Point<dim> &, std::vector<CaptureBox<dim>> &);

        void find_absolute_closest(const dealii::Point<dim> &, std::vector<CaptureBox<dim>> &, std::vector<dealii::CellId> &);

        double calculate_min_gap(dealii::Point<dim> &, const dealii::Point<dim> &, 
        const std::vector<dealii::CellId> &, bool);    
      
    };


    //////////////////////////// SKULL inline fxs ////////////////////////////


    template <int dim>
    void
    Skull<dim>::
    declare_parameters (dealii::ParameterHandler &prm)
    {   
    prm.declare_entry("boundary file","",dealii::Patterns::FileName());

    efilog(Verbosity::verbose) << "Skull obstacle finished "
                                  "declaring parameters."
                               << std::endl;
    }

    template <int dim>
    void
    Skull<dim>::
    parse_parameters (dealii::ParameterHandler &prm)
    {   
        auto boundary_filename = prm.get ("boundary file");

        boost::filesystem::path input_directory = 
                GlobalParameters::get_input_directory();

        // input directory
        std::string directory (input_directory.string()
                            + std::string(1,input_directory.separator));

        // common name of the output files
        std::string name (directory + boundary_filename);
        this->set_boundary_file(name);

        efilog(Verbosity::verbose) << "Skull obstacle finished "
                                  "parsing parameters."
                               << std::endl;
    }


    //////////////////////////// CaptureBox inline fxs ////////////////////////////

    template<int dim>
    CaptureBox<dim>::CaptureBox(dealii::BoundingBox<dim> box, dealii::CellId cellID)
    : boxDimensions(box),
    cell_id(cellID)
    {}

    template<int dim>
    double CaptureBox<dim>::upperLimit(const int comp) const
    {
        return this->boxDimensions.upper_bound(comp);
    }

    template<int dim>
    double CaptureBox<dim>::lowerLimit(const int comp) const
    {
        return this->boxDimensions.lower_bound(comp);
    }

    template<int dim>
    dealii::CellId CaptureBox<dim>::getCellId() const
    {
        return this->cell_id;
    }

    //////////////////////////// CollectionBoxes inline fxs ////////////////////////////
    
    template<int dim>
    CollectionBoxes<dim>::CollectionBoxes(int comp)
    : componentOrderedOn(comp)
    , maxVal(-1.*std::numeric_limits<double>::max())
    , minVal(std::numeric_limits<double>::max())
    {}

    template<int dim>
    void CollectionBoxes<dim>::addBox(CaptureBox<dim> & box)
    {
        this->maxVal = box.upperLimit(this->componentOrderedOn) > this->maxVal ? 
            box.upperLimit(this->componentOrderedOn) : this->maxVal;
        this->minVal = box.lowerLimit(this->componentOrderedOn) < this->minVal ? 
            box.lowerLimit(this->componentOrderedOn) : this->minVal;
        captureBoxes.push_back(box);
    }

    template<int dim>
    double CollectionBoxes<dim>::getMaxVal() const
    {
        return this->maxVal;
    }

    template<int dim>
    double CollectionBoxes<dim>::getMinVal() const
    {
        return this->minVal;
    }

    template<int dim>
    std::vector<CaptureBox<dim>> & CollectionBoxes<dim>::getCaptureBoxes()
    {
        return this->captureBoxes;
    }

    template<int dim>
    void CollectionBoxes<dim>::setNextCoordBST(BST<dim> * bst)
    {
        this->nextCoordBST = bst;
    }

    template<int dim>
    BST<dim>* CollectionBoxes<dim>::getNextCoordBST() const
    {
        return this->nextCoordBST;
    }

    template<int dim>
    int CollectionBoxes<dim>::size() const
    {
        return captureBoxes.size();
    }

    //////////////////////////// Node inline fxs ////////////////////////////
    template<int dim>
    Node<dim>::Node(double cuttingLine)
    : cuttingLine(cuttingLine)
    {
    this->left = nullptr;
    this->right = nullptr;
    }

    template<int dim>
    void Node<dim>::addBox(CollectionBoxes<dim> & collectionBox)
    {
        collectionBoxes.push_back(collectionBox);
    }

    template<int dim>
    void Node<dim>::addBoxes(std::vector<CollectionBoxes<dim>> & newCollectionBoxes)
    {
        
        this->collectionBoxes.insert(collectionBoxes.begin(), newCollectionBoxes.begin(), newCollectionBoxes.end());
    }

    template<int dim>
    std::vector<CollectionBoxes<dim>> Node<dim>::getCollectionBoxes()
    {
        return this->collectionBoxes;
    }

    template<int dim>
    double Node<dim>::getCuttingLine()
    {
        return this->cuttingLine;
    }

    template<int dim>
    Node<dim>* Node<dim>::getLeft() const
    {
        return this->left;
    }

    template<int dim>
    Node<dim>* Node<dim>::getRight() const
    {
        return this->right;
    }

    template<int dim>
    void Node<dim>::setLeft(Node<dim>* newNode)
    {
        this->left = newNode;
    }

    template<int dim>
    void Node<dim>::setRight(Node<dim>* newNode)
    {
        this->right = newNode;
    }
    //////////////////////////// BST inline fxs ////////////////////////////
    template<int dim>
    BST<dim>::BST(int comp)
    :sortedComp(comp)
    {
    this->root = nullptr;
    }

    template<int dim>
    void BST<dim>::search(const dealii::Point<dim> & queryPoint, int* comp, std::vector<CaptureBox<dim>> & result)
    {
    Node<dim>* currentNode = this->root;
    while( currentNode->getCuttingLine() != queryPoint(*comp))
    {
        double cuttingLine = currentNode->getCuttingLine();
        // std::cout << "Current value: " << cuttingLine << std::endl;
        std::vector<CollectionBoxes<dim>> collectionBoxes = currentNode->getCollectionBoxes();
        for (auto boxIt = collectionBoxes.begin(); boxIt != collectionBoxes.end(); boxIt++)
        {
        if ( ((queryPoint(*comp) - boxIt->getMinVal()) >= -1e-6) &&
                ((queryPoint(*comp) - boxIt->getMaxVal()) <= 1e-6) )
                {
                int* newComp = new int(*comp+1);
                BST<dim>* nextBST = boxIt->getNextCoordBST();
                if (nextBST != nullptr)
                    nextBST->search(queryPoint, newComp, result);
                else
                {
                    std::vector<CaptureBox<dim>> boxesToAdd =  boxIt->getCaptureBoxes();  
                    result.insert(result.end(), boxesToAdd.begin(), boxesToAdd.end());
                }
                }    
        }
        if (cuttingLine - queryPoint(*comp) >= -1e-6)
        {
        currentNode = currentNode->getLeft();
        if (currentNode == nullptr)
            return;
        }
        else
        {
        currentNode = currentNode->getRight();
        if (currentNode == nullptr)
            return;
        }
    }

    }

    template<int dim>
    Node<dim>* BST<dim>::getRoot()
    {
    return this->root;
    }

    template<int dim>
    void BST<dim>::insert(Node<dim>& newNode)
    {
        // std::cout << "BST insert started"<< std::endl;
        // std::cout << "newNode.getCuttingLine(): " << newNode.getCuttingLine() << std::endl;
        if (this->root == nullptr)
        {
            this->root = &newNode;
            return;
        }
        Node<dim>* current = this->root;
        while(true)
        {
            Node<dim>* parent  = current;
            //  std::cout << "current->getCuttingLine(): " << current->getCuttingLine() << std::endl;
            if ( (newNode.getCuttingLine() - current->getCuttingLine()) < 1e-8) // Go left
            {
                current = current->getLeft();
                if (current == nullptr)
                {
                // std::cout << "\t INSERTED left"<< std::endl;
                    parent->setLeft(&newNode);
                    return;
                }
                // std::cout << "\t SEARCHING left"<< std::endl;
            } else
            {
                current = current->getRight();
                if (current == nullptr)
                {
                // std::cout << "\t INSERTED right"<< std::endl;
                    parent->setRight(&newNode);
                    return;
                }
                // std::cout << "\t SEARCHING right"<< std::endl;
            }
        }
    }
} // efi namespace
#endif