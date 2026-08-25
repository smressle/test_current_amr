//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file default_pgen.cpp
//! \brief Provides default (empty) versions of all functions in problem generator files
//! This means user does not have to implement these functions if they are not needed.
//!
//! The attribute "weak" is used to ensure the loader selects the user-defined version of
//! functions rather than the default version given here.
//!
//! The attribute "alias" may be used with the "weak" functions (in non-defining
//! declarations) in order to have them refer to common no-operation function definition
//! in the same translation unit. Target function must be specified by mangled name
//! unless C linkage is specified.
//!
//! This functionality is not in either the C nor the C++ standard. These GNU extensions
//! are largely supported by LLVM, Intel, IBM, but may affect portability for some
//! architecutres and compilers. In such cases, simply define all 6 of the below class
//! functions in every pgen/*.cpp file (without any function attributes).

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../eos/eos.hpp"

// 3x members of Mesh class:

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void __attribute__((weak)) Mesh::InitUserMeshData(ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void __attribute__((weak)) Mesh::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Function called after main loop is finished for user-defined work.
//========================================================================================

void __attribute__((weak)) Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // do nothing
  return;
}

// 4x members of MeshBlock class:

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void __attribute__((weak)) MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================

void __attribute__((weak)) MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // In practice, this function should *always* be replaced by a version
  // that sets the initial conditions for the problem of interest.
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Function called before generating output files
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // do nothing
  return;
}

void __attribute__((weak)) HydroSourceTerms::ApplyBondiBoundaries(Real time,MeshBlock *pmb, AthenaArray<Real> &cons, 
           const AthenaArray<Real> &prim_old, FaceField &b,AthenaArray<Real> &prim, AthenaArray<Real> &bcc, 
           Coordinates *pco){
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Function called before generating output files
//========================================================================================

void __attribute__((weak)) EquationOfState::GetRadii(Real t, Real x1, Real x2, Real x3, Real a, Real *r1, Real *r2) {

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") == 0){
    (*r1) =std::sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  }
  else if ( (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) or (std::strcmp(COORDINATE_SYSTEM, "schwarzschild") == 0) or (std::strcmp(COORDINATE_SYSTEM, "kerr-schild") == 0) ){
    (*r1) = x1;
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0){
    (*r1) =std::sqrt( SQR(x1) + SQR(x3));
  }
  else if (std::strcmp(COORDINATE_SYSTEM, "gr_user") == 0){
      Real x = x1;
      Real y = x2;
      Real z = x3;
      Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
      Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
      (*r1)=r;
      (*r2)=-1.0;
  }
  else (*r1) = std::abs(x1);

  (*r2)=-1.0;
  return;
}
// Real __attribute__((weak)) EquationOfState::GetRadius2(Real t, Real x1, Real x2, Real x3) {
//   // do nothing
//   return -1.0;
// }


void __attribute__((weak)) Mesh::get_bh_positions( Real t, Real *xbh1,Real *ybh1,Real *zbh1, Real *xbh2,Real *ybh2,Real *zbh2){


  *xbh1 = 0.0;
  *xbh2 = 0.0;

  *ybh1 = 0.0;
  *ybh2 = 0.0;

  *zbh1 = 0.0;
  *zbh2 = 0.0;

  return;

}
void __attribute__((weak)) Mesh::get_prime_coords_wrapper(int BH_INDEX, Real t, Real x, Real y, Real z, 
                                  Real *x_prime, Real *y_prime, Real *z_prime){


  *x_prime = 0.0;
  *y_prime = 0.0;
  *z_prime = 0.0;

  return;

}
void __attribute__((weak)) Mesh::boost_lowered_vector_wrapper( int BH_INDEX, Real t, Real u0, Real u1, Real u2, Real u3, 
                                Real *u0_prime, Real *u1_prime, Real *u2_prime, Real *u3_prime){


  *u0_prime = 0.0;
  *u1_prime = 0.0;
  *u2_prime = 0.0;
  *u3_prime = 0.0;

  return;

}