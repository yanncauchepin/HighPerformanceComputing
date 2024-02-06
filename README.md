## TO DO LIST ##

### PowerMethod ###

- [ ] Correct and MAJ reports.
- [ ] Upgrade CUDA version to allow *atomicAdd* to be applied to double type.
- [ ] CUDA : Check old vs new MAJ of X and Y at the end of each iteration.
- [ ] Bug CUDA : error still infinite along iteration.
- [ ] Bug MPI : error still constant after several iteration.
- [ ] Check the good initialization of X ; if we must apply size or size*size.
- [ ] MPI : when divide block_size, handle case size/number_processes = int + decimal where decimal non null
- [ ] MPI : Optimize reading set_flattened_submatrix_from_file : error in pointer fseek depending on decimal of double values for example.
- [ ] MPI : Add alternative to MPI commands and check optimization, maybe like *Ssend* for example
- [ ] Upgrade README.txt to add metadata of input for PowerMethod.

### Mandelbrot ###

- [ ] Restart all codes and upgrade repository.
- [ ] Correct and MAJ reports.

### Cholesky ###

- [ ] Correct and MAJ reports.
- [ ] Check for upgrade, maybe create a generic code.

### MPI ###

- [ ] Choose what to do with those files.
- [ ] MAJ and improve files.
