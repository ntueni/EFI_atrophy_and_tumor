
## Installation

The efi library requires deal.II 9.1.1 to be installed with trilinos enabled. It is recommended to install dealii via spack. Further, the boost libraries *filesystem*, *tti*, and *any* are required. 

1. Make project directory, e.g. *myproject*.
2. Enter the directory.
3. Copy the src folder to the project directory.
4. Create a build directory.
5. Enter the build directory.
6. Run cmake (out-of-source-build).

To do so, run the following comands in a terminal:
<pre><code>
spack load dealii@...
spack load boost@...
spack load openmpi@...
mkdir myproject
cd myproject
cp -a /path/to/src .
mkdir build
cd /build 
cmake ../src
make debug
</code></pre>
or alternatively
<pre><code>
make release
</code></pre>

## Runnig the executable
The executable can be run form the terminal via
<pre><code>
mpirun -np 1 --bind-to socket efi_vlab 3 myparams.prm
</code></pre>
where the number after <code>-np</code> specifies the number of processors, the argument after <code>--bind-to</code> specifies the binding pattern, <code>efi_vlab</code> is the application we want to run (short for emerging fields initiative virtual laboratory). The following two arguments specify the number of threads per process and the input paramter file, here <code>3</code> and <code>myparams</code>, respectively.

## Eclipse support
Instead of running
<pre><code>
cmake ../src
</code></pre>
use the command 
<pre><code>
cmake -D_ECLIPSE_VERSION=4.8 -G"Eclipse CDT4 - Unix Makefiles" ../src/
</code></pre>
instead. 
Now, open eclipse and import the folder *myproject* choosing "Existing Code as Makefile Project". Note that if you add header files to src/include everything is fine, but if you add new source files you have to run 
<code>
$cmake ../src
</code>
from the build folder, otherwise the makefile won't be updated and the new source file will be ignored. Make sure that you are not using the flags <code>-D_ECLIPSE_VERSION=4.8 -G"Eclipse CDT4 - Unix Makefiles"</code> this time.

**Adding bulid targets**

In the build tasks tab in the eclipse editor, klick on your project, choose the build folder and create the targets "debug", "release", and "clean". These targets do exactly what their names say, building in debug/release mode, or cleaning the make-output.

