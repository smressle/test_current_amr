//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_bondi.cpp
//  \brief Problem generator for spherically symmetric black hole accretion.

// C++ headers
#include <cmath>  // abs(), NAN, pow(), sqrt()

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>


// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../utils/utils.hpp" //ran2()


// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);


void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);

int RefinementCondition(MeshBlock *pmb);

// Global variables
static Real m, a;          // black hole mass and spin
static Real dfloor,pfloor;                         // density and pressure floors

static Real rh;
static Real r_cut;        // initial condition cut off
static Real bondi_radius;  // b^2/rho at inner radius
static Real field_norm;  
static Real magnetic_field_inclination;   
Real x1_harm_max;  //maximum x1 coordinate for hyper-exponentiation   
Real cpow2,npow2;  //coordinate parameters for hyper-exponentiation
Real rbr; //break radius for hyper-exp.

int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
int max_smr_level = 0;
Real r_inner_boundary = 0.0;


//----------------------------------------------------------------------------------------
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters

  bondi_radius  = pin->GetReal("problem", "bondi_radius");

  r_cut = pin->GetReal("problem", "r_cut");
  field_norm =  pin->GetReal("problem", "field_norm");

  if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


  magnetic_field_inclination = pin->GetOrAddReal("problem","field_inclination",0.0);

  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, FixedBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, FixedBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, FixedBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, FixedBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, FixedBoundary);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, FixedBoundary);

  EnrollUserMetric(Cartesian_GR);

  // EnrollUserRadSourceFunction(inner_boundary);



  return;
}

//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)
// Notes:
//   user arrays are metric and its inverse

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  int N_user_vars = 6;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }

  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);


    // Get mass of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();

  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

  max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);
  max_smr_level = pin->GetOrAddReal("mesh","smrlevel",max_refinement_level); 
  if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;


  return;
}


    static Real exp_cut_off(Real r){

      if (r<=rh) return 0.0;
      else if (r<= r_cut) return std::exp(5 * (r-r_cut)/r);
      else return 1.0;
    }

    static Real Ax_func(Real x,Real y, Real z){

      return 0.0 * field_norm;
    }
    static Real Ay_func(Real x, Real y, Real z){
      return (-z * std::sin(magnetic_field_inclination) + x * std::cos(magnetic_field_inclination) ) * field_norm;  //x 
    }
    static Real Az_func(Real x, Real y, Real z){
      return 0.0 * field_norm;
    }
//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets primitive and conserved variables according to input primitives
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Parameters
  Real cs_0 = 1.0/std::sqrt(bondi_radius);
  Real rho_0 = 1.0;
  Real gam = peos->GetGamma();
  Real P_0 = SQR(cs_0)*rho_0/gam;

  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );



  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

// Ensure a different initial random seed for each meshblock.
  int64_t iseed = -1 - gid;

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
            &r, &theta, &phi);
        Real u0 = std::sqrt(-1.0/g(I00,i));
        Real uu1 = 0.0 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = 0.0 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = 0.0 - gi(I03,i)/gi(I00,i) * u0;

        Real amp = 0.00;
        //if (std::fabs(a)<1e-1) amp = 0.01;
        Real rval = amp*(ran2(&iseed) - 0.5);
    
        if (r<r_cut){
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = 0.0;
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = 0.0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
        }
        else{ 
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho_0 * (1.0 + 2.0*rval);
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = P_0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
      }
    }
  }

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {


    Real delta =0.0; // 1e-1;  //perturbation in B-field amplitude
    Real pert = 0.0;


    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);

    AthenaArray<Real> A3,A1,A2;

    A1.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A2.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A3.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );

      // Set B^1
      for (int k = kl; k <= ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {

            //A1 defined at cell center in x1 but face in x2 x3, 
            //A2 defined at cell center in x2 but face in x1 x3,
            //A3 defined at cell center in x3 but face in x1 x2

            Real r,theta,phi;
            Real x_coord;
            if (i<= iu) x_coord = pcoord->x1v(i);
            else x_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);
            GetBoyerLindquistCoordinates(x_coord,pcoord->x2f(j),pcoord->x3f(k), &r, &theta,&phi);

            pert = delta * std::cos(phi);
            Real x,y,z;
            GetCKSCoordinates(x_coord,pcoord->x2f(j),pcoord->x3f(k),&x,&y,&z);
            Real Ax = Ax_func(x,y,z) * (1 + pert);
            Real Ay = Ay_func(x,y,z) * (1 + pert);
            Real Az = Az_func(x,y,z) * (1 + pert);

            // Real Ar,Ath,Aphi,A0;;

            // TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A1(k,j,i) = Ax * exp_cut_off(r);

            Real y_coord;
            if (j<= ju) y_coord = pcoord->x2v(j);
            else y_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),y_coord,pcoord->x3f(k), &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(pcoord->x1f(i),y_coord,pcoord->x3f(k),&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            //TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A2(k,j,i) = Ay * exp_cut_off(r);

            Real z_coord;
            if (k<= ku) z_coord = pcoord->x3v(k);
            else z_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),pcoord->x2f(j),z_coord, &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(pcoord->x1f(i),pcoord->x2f(j),z_coord,&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            //TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            A3(k,j,i) = Az * exp_cut_off(r);



            }
          }
        }


      // Initialize interface fields
    AthenaArray<Real> area;
    area.NewAthenaArray(ncells1+1);

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = -1.0*(pcoord->dx3f(k)*A3(k,j,i+1) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
          if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
          //if (j==ju) fprintf(stderr,"B: %g area: %g theta: %g j: %d A3: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
           // A3(k,j,i+1), A3(k,j,i));

          if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face3Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = (pcoord->dx2f(j)*A2(k,j,i+1) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
          //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;

          if (std::isnan((pfield->b.x3f(k,j,i)))){

           fprintf(stderr,"isnan in bx3!\n A2: %g %g \n area: %g dx2f: %g \n", A2(k,j,i+1),A2(k,j,i),area(i),pcoord->dx2f(j));
           exit(0);
         }
        }
      }
    }

    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) = (pcoord->dx3f(k)*A3(k,j+1,i) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face3Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x3f(k,j,i) -= (pcoord->dx1f(i)*A1(k,j+1,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x3f(k,j,i)))) {
              fprintf(stderr,"isnan in bx3!\n A1: %g %g \n area: %g dx1f: %g \n", A1(k,j+1,i),A1(k,j,i),area(i),pcoord->dx1f(i));
              exit(0);
            }
          }
        }
      }
    }
    // for 3D only
    if (block_size.nx3 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) -= (pcoord->dx2f(j)*A2(k+1,j,i) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x2f(k,j,i) += (pcoord->dx1f(i)*A1(k+1,j,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
            if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
            //if ( ju==je && j==je) fprintf(stderr,"B: %g area: %g theta: %g j: %d A1: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
            //A1_bound(k+1,j,i), A1_bound(k,j,i));
          }
        }
      }
    }

    area.DeleteAthenaArray();
    A1.DeleteAthenaArray();
    A2.DeleteAthenaArray();
    A3.DeleteAthenaArray();


  }


  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
      pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
        kl, ku);
  } else {
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
    bb.DeleteAthenaArray();
  }



  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();

    // Call user work function to set output variables
  UserWorkInLoop();
  return;
}

//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  return;
}

//----------------------------------------------------------------------------------------
// Inflow boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // Set hydro variables
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,is);
        prim(IEN,k,j,i) = prim(IEN,k,j,is);
        prim(IM1,k,j,i) = std::min(prim(IM1,k,j,is), static_cast<Real>(0.0));
        prim(IM2,k,j,i) = prim(IM2,k,j,is);
        prim(IM3,k,j,i) = prim(IM3,k,j,is);
      }
    }
  }
  if (not MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // Set radial magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x1f(k,j,i) = bb.x1f(k,j,is);
      }
    }
  }

  // Set polar magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je+1; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x2f(k,j,i) = bb.x2f(k,j,is);
      }
    }
  }

  // Set azimuthal magnetic field
  for (int k = ks; k <= ke+1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x3f(k,j,i) = bb.x3f(k,j,is);
      }
    }
  }
  return;
}

/* 
Simple function to get Cartesian Coordinates

*/
void get_cartesian_coords(const Real x1, const Real x2, const Real x3, Real *x, Real *y, Real *z){


  if (COORDINATE_SYSTEM == "cartesian"){
      *x = x1;
      *y = x2;
      *z = x3;
    }
    else if (COORDINATE_SYSTEM == "cylindrical"){
      *x = x1*std::cos(x2);
      *y = x1*std::sin(x2);
      *z = x3;
  }
    else if (COORDINATE_SYSTEM == "spherical_polar"){
      *x = x1*std::sin(x2)*std::cos(x3);
      *y = x1*std::sin(x2)*std::sin(x3);
      *z = x1*std::cos(x2);
    }

}
/*
Returns approximate cell sizes if the grid was uniform
*/

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ){

  if (COORDINATE_SYSTEM == "cartesian" || COORDINATE_SYSTEM == "gr_user"){
    *DX = (box_size.x1max-box_size.x1min)/(1. * box_size.nx1);
    *DY = (box_size.x2max-box_size.x2min)/(1. * box_size.nx2);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);

  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
  }
}


int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);


  Real total_box_radius = (pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min)/2.0;
  Real bh2_focus_radius = 3.125;
  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  if (current_level >=max_refinement_level) return 0;

  int any_at_current_level=0;

  int max_second_bh_refinement_level=max_refinement_level;

  // fprintf(stderr,"current level: %d max_refinement_level: %d max_smr_refinement: %d max_bh2_refinement: %d \n",current_level,max_refinement_level,max_smr_refinement_level,max_second_bh_refinement_level);
  //first loop: check if any part of block is within refinement levels for secondary black hole

  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {


          for (int n_level = 1; n_level<=max_second_bh_refinement_level; n_level++){
          
            Real x = pmb->pcoord->x1v(i);
            Real y = pmb->pcoord->x2v(j);
            Real z = pmb->pcoord->x3v(k);

            Real xprime = x;
            Real yprime = y;
            Real zprime = z-20.0;

            Real box_radius = bh2_focus_radius * std::pow(2.,max_second_bh_refinement_level - n_level)*0.9999;

        
            //           if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
            // fprintf(stderr,"current level (AMR): %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
            // }
            if (xprime < box_radius && xprime > -box_radius && yprime < box_radius
              && yprime > -box_radius && zprime < box_radius && zprime > -box_radius ){
              if (current_level < n_level){

                  return  1;
              }
            }

            if (current_level==n_level) any_at_current_level=1;

          
          }

        }
      }
    }
      


if (any_at_current_level==1) return 0;
  return -1;
}


/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


//   Real r,th,ph;
//   AthenaArray<Real> g, gi;
//   g.InitWithShallowCopy(pmb->ruser_meshblock_data[0]);
//   gi.InitWithShallowCopy(pmb->ruser_meshblock_data[1]);



//    for (int k=pmb->ks; k<=pmb->ke; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=pmb->js; j<=pmb->je; ++j) {
//       pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
// #pragma simd
//       for (int i=pmb->is; i<=pmb->ie; ++i) {


//          GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

//           if (r < r_inner_boundary){
              

//               //set uu assuming u is zero
//               Real gamma = 1.0;
//               Real alpha = std::sqrt(-1.0/gi(I00,i));
//               Real u0 = gamma/alpha;
//               Real uu1 = - gi(I01,i)/gi(I00,i) * u0;
//               Real uu2 = - gi(I02,i)/gi(I00,i) * u0;
//               Real uu3 = - gi(I03,i)/gi(I00,i) * u0;
              
//               prim(IDN,k,j,i) = dfloor;
//               prim(IVX,k,j,i) = 0.;
//               prim(IVY,k,j,i) = 0.;
//               prim(IVZ,k,j,i) = 0.;
//               prim(IPR,k,j,i) = pfloor;
            
              
              
//           }



// }}}

// g.DeleteAthenaArray();
// gi.DeleteAthenaArray();



}
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k, kprime;
  int is, ie, js, je, ks, ke;


  apply_inner_boundary_condition(pmb,prim);

  return;
}
//----------------------------------------------------------------------------------------
// Function responsible for storing useful quantities for output
// Inputs: (none)
// Outputs: (none)
// Notes:
//   writes to user_out_var array the following quantities:
//     0: gamma (normal-frame Lorentz factor)
//     1: p_mag (magnetic pressure)
// void MeshBlock::UserWorkInLoop(void)
// {
//   // Create aliases for metric
//   AthenaArray<Real> g, gi;
//   g.InitWithShallowCopy(ruser_meshblock_data[0]);
//   gi.InitWithShallowCopy(ruser_meshblock_data[1]);


//   // Go through all cells
//   for (int k = ks; k <= ke; ++k) {
//     for (int j = js; j <= je; ++j) {
//       pcoord->CellMetric(k, j, is, ie, g, gi);
//       for (int i = is; i <= ie; ++i) {

//         // Calculate normal frame Lorentz factor
//         Real uu1 = phydro->w(IM1,k,j,i);
//         Real uu2 = phydro->w(IM2,k,j,i);
//         Real uu3 = phydro->w(IM3,k,j,i);
//         Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
//                  + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
//                  + g(I33,i)*uu3*uu3;
//         Real gamma = std::sqrt(1.0 + tmp);
//         user_out_var(0,k,j,i) = gamma;

//         // Calculate 4-velocity
//         Real alpha = std::sqrt(-1.0/gi(I00,i));
//         Real u0 = gamma/alpha;
//         Real u1 = uu1 - alpha * gamma * gi(I01,i);
//         Real u2 = uu2 - alpha * gamma * gi(I02,i);
//         Real u3 = uu3 - alpha * gamma * gi(I03,i);
//         Real u_0, u_1, u_2, u_3;

//         user_out_var(1,k,j,i) = u0;
//         user_out_var(2,k,j,i) = u1;
//         user_out_var(3,k,j,i) = u2;
//         user_out_var(4,k,j,i) = u3;
//         if (not MAGNETIC_FIELDS_ENABLED) {
//           continue;
//         }

//         pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

//         // Calculate 4-magnetic field
//         Real bb1 = pfield->bcc(IB1,k,j,i);
//         Real bb2 = pfield->bcc(IB2,k,j,i);
//         Real bb3 = pfield->bcc(IB3,k,j,i);
//         Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
//                 + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
//                 + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
//                 + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
//         Real b1 = (bb1 + b0 * u1) / u0;
//         Real b2 = (bb2 + b0 * u2) / u0;
//         Real b3 = (bb3 + b0 * u3) / u0;
//         Real b_0, b_1, b_2, b_3;
//         pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

//         // Calculate magnetic pressure
//         Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
//         user_out_var(5,k,j,i) = b_sq/2.0;
//       }
//     }
//   }
//   return;
// }
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems


static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi) {

    Real x = x1;
    Real y = x2;
    Real z = x3;
    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(z/r);
    *pphi = std::atan2( (r*y-a*x)/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );
  return;
}
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z){
  *x = r;
  *y = th ;
  *z = phi; 

}
//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
  return;
}

static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

    *pa0 = a0_cks ; 
    *pa1 =         (std::sin(theta)*std::cos(phi)) * a1_cks 
                 + (std::sin(theta)*std::sin(phi)) * a2_cks 
                 + (std::cos(theta)              ) * a3_cks ;

    *pa2 =         (std::cos(theta) * (r*std::cos(phi) + a*std::sin(phi) ) ) * a1_cks
                 + (std::cos(theta) * (r*std::sin(phi) - a*std::cos(phi) ) ) * a2_cks
                 + (-r*std::sin(theta)                                     ) * a3_cks;

    *pa3 =          -y * a1_cks 
                + (x) * a2_cks;


  return;
}



void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  a = pin->GetReal("coord", "a");
  Real a_spin =a;

  Real SMALL = 1e-5;
  if (std::fabs(z)<SMALL) z= SMALL;
  Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = std::sqrt(r/2.0);


  //if (r<0.01) r = 0.01;


  Real eta[4],l_lower[4],l_upper[4];

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  l_upper[0] = -1.0;
  l_upper[1] = (r*x + a_spin*y)/( SQR(r) + SQR(a) );
  l_upper[2] = (r*y - a_spin*x)/( SQR(r) + SQR(a) );
  l_upper[3] = z/r;

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;





  // Set covariant components
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0];
  g(I01) = f * l_lower[0]*l_lower[1];
  g(I02) = f * l_lower[0]*l_lower[2];
  g(I03) = f * l_lower[0]*l_lower[3];
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1];
  g(I12) = f * l_lower[1]*l_lower[2];
  g(I13) = f * l_lower[1]*l_lower[3];
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2];
  g(I23) = f * l_lower[2]*l_lower[3];
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3];




  // // Set contravariant components
  g_inv(I00) = eta[0] - f * l_upper[0]*l_upper[0];
  g_inv(I01) = -f * l_upper[0]*l_upper[1];
  g_inv(I02) = -f * l_upper[0]*l_upper[2];
  g_inv(I03) = -f * l_upper[0]*l_upper[3];
  g_inv(I11) = eta[1] - f * l_upper[1]*l_upper[1];
  g_inv(I12) = -f * l_upper[1]*l_upper[2];
  g_inv(I13) = -f * l_upper[1]*l_upper[3];
  g_inv(I22) = eta[2] -f * l_upper[2]*l_upper[2];
  g_inv(I23) = -f * l_upper[2]*l_upper[3];
  g_inv(I33) = eta[3] - f * l_upper[3]*l_upper[3];


  Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);
  Real rsq_p_asq = SQR(r) + SQR(a);

  Real df_dx1 = SQR(f)*x/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a_spin*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
  Real df_dx2 = SQR(f)*y/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a_spin*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  Real df_dx3 = SQR(f)*z/(2.0*std::pow(r,5)) * ( ( ( 3.0*SQR(a_spin*z)-SQR(r)*SQR(r) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(a*r)) ;
  //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a_spin/( rsq_p_asq );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a_spin/( rsq_p_asq );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
  Real dl3_dx1 = - x*z/(r) /( sqrt_term );
  Real dl3_dx2 = - y*z/(r) /( sqrt_term );
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

  Real dl0_dx1 = 0.0;
  Real dl0_dx2 = 0.0;
  Real dl0_dx3 = 0.0;

  if (std::isnan(f) || std::isnan(r) || std::isnan(sqrt_term) || std::isnan (df_dx1) || std::isnan(df_dx2)){
    fprintf(stderr,"ISNAN in metric\n x y y: %g %g %g r: %g \n",x,y,z,r);
    exit(0);
  }




  // Set x-derivatives of covariant components
  dg_dx1(I00) = df_dx1*l_lower[0]*l_lower[0] + f * dl0_dx1 * l_lower[0] + f * l_lower[0] * dl0_dx1;
  dg_dx1(I01) = df_dx1*l_lower[0]*l_lower[1] + f * dl0_dx1 * l_lower[1] + f * l_lower[0] * dl1_dx1;
  dg_dx1(I02) = df_dx1*l_lower[0]*l_lower[2] + f * dl0_dx1 * l_lower[2] + f * l_lower[0] * dl2_dx1;
  dg_dx1(I03) = df_dx1*l_lower[0]*l_lower[3] + f * dl0_dx1 * l_lower[3] + f * l_lower[0] * dl3_dx1;
  dg_dx1(I11) = df_dx1*l_lower[1]*l_lower[1] + f * dl1_dx1 * l_lower[1] + f * l_lower[1] * dl1_dx1;
  dg_dx1(I12) = df_dx1*l_lower[1]*l_lower[2] + f * dl1_dx1 * l_lower[2] + f * l_lower[1] * dl2_dx1;
  dg_dx1(I13) = df_dx1*l_lower[1]*l_lower[3] + f * dl1_dx1 * l_lower[3] + f * l_lower[1] * dl3_dx1;
  dg_dx1(I22) = df_dx1*l_lower[2]*l_lower[2] + f * dl2_dx1 * l_lower[2] + f * l_lower[2] * dl2_dx1;
  dg_dx1(I23) = df_dx1*l_lower[2]*l_lower[3] + f * dl2_dx1 * l_lower[3] + f * l_lower[2] * dl3_dx1;
  dg_dx1(I33) = df_dx1*l_lower[3]*l_lower[3] + f * dl3_dx1 * l_lower[3] + f * l_lower[3] * dl3_dx1;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = df_dx2*l_lower[0]*l_lower[0] + f * dl0_dx2 * l_lower[0] + f * l_lower[0] * dl0_dx2;
  dg_dx2(I01) = df_dx2*l_lower[0]*l_lower[1] + f * dl0_dx2 * l_lower[1] + f * l_lower[0] * dl1_dx2;
  dg_dx2(I02) = df_dx2*l_lower[0]*l_lower[2] + f * dl0_dx2 * l_lower[2] + f * l_lower[0] * dl2_dx2;
  dg_dx2(I03) = df_dx2*l_lower[0]*l_lower[3] + f * dl0_dx2 * l_lower[3] + f * l_lower[0] * dl3_dx2;
  dg_dx2(I11) = df_dx2*l_lower[1]*l_lower[1] + f * dl1_dx2 * l_lower[1] + f * l_lower[1] * dl1_dx2;
  dg_dx2(I12) = df_dx2*l_lower[1]*l_lower[2] + f * dl1_dx2 * l_lower[2] + f * l_lower[1] * dl2_dx2;
  dg_dx2(I13) = df_dx2*l_lower[1]*l_lower[3] + f * dl1_dx2 * l_lower[3] + f * l_lower[1] * dl3_dx2;
  dg_dx2(I22) = df_dx2*l_lower[2]*l_lower[2] + f * dl2_dx2 * l_lower[2] + f * l_lower[2] * dl2_dx2;
  dg_dx2(I23) = df_dx2*l_lower[2]*l_lower[3] + f * dl2_dx2 * l_lower[3] + f * l_lower[2] * dl3_dx2;
  dg_dx2(I33) = df_dx2*l_lower[3]*l_lower[3] + f * dl3_dx2 * l_lower[3] + f * l_lower[3] * dl3_dx2;

  // Set phi-derivatives of covariant components
  dg_dx3(I00) = df_dx3*l_lower[0]*l_lower[0] + f * dl0_dx3 * l_lower[0] + f * l_lower[0] * dl0_dx3;
  dg_dx3(I01) = df_dx3*l_lower[0]*l_lower[1] + f * dl0_dx3 * l_lower[1] + f * l_lower[0] * dl1_dx3;
  dg_dx3(I02) = df_dx3*l_lower[0]*l_lower[2] + f * dl0_dx3 * l_lower[2] + f * l_lower[0] * dl2_dx3;
  dg_dx3(I03) = df_dx3*l_lower[0]*l_lower[3] + f * dl0_dx3 * l_lower[3] + f * l_lower[0] * dl3_dx3;
  dg_dx3(I11) = df_dx3*l_lower[1]*l_lower[1] + f * dl1_dx3 * l_lower[1] + f * l_lower[1] * dl1_dx3;
  dg_dx3(I12) = df_dx3*l_lower[1]*l_lower[2] + f * dl1_dx3 * l_lower[2] + f * l_lower[1] * dl2_dx3;
  dg_dx3(I13) = df_dx3*l_lower[1]*l_lower[3] + f * dl1_dx3 * l_lower[3] + f * l_lower[1] * dl3_dx3;
  dg_dx3(I22) = df_dx3*l_lower[2]*l_lower[2] + f * dl2_dx3 * l_lower[2] + f * l_lower[2] * dl2_dx3;
  dg_dx3(I23) = df_dx3*l_lower[2]*l_lower[3] + f * dl2_dx3 * l_lower[3] + f * l_lower[2] * dl3_dx3;
  dg_dx3(I33) = df_dx3*l_lower[3]*l_lower[3] + f * dl3_dx3 * l_lower[3] + f * l_lower[3] * dl3_dx3;


  return;
}

