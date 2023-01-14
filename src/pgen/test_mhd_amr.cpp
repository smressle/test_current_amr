/*
 * Function star_wind.c
 *
 * Problem generator for stars with solar wind output, with gravity included
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

#include <iostream>


#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"


/* cooling */
/* -------------------------------------------------------------------------- */
static int cooling;
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );
static Real Lambda_T(const Real T);
static Real Yinv(Real Y1);
static Real Y(const Real T);
static Real tcool(const Real d, const Real T);



Real mp_over_kev = 9.994827;   //mp * (pc/kyr)^2/kev
Real UnitDensity = 6.767991e-23; // solar mass pc^-3
Real UnitEnergyDensity = 6.479592e-7; //solar mass /(pc ky^2)
Real UnitTime = 3.154e10;  //kyr
Real Unitlength = 3.086e+18; //parsec
Real UnitB = Unitlength/UnitTime * std::sqrt(4. * PI* UnitDensity);
Real UnitLambda_times_mp_times_kev = 1.255436328493696e-21 ;//  UnitEnergyDensity/UnitTime*Unitlength**6.*mp*kev/(solar_mass**2. * Unitlength**2./UnitTime**2.)
Real keV_to_Kelvin = 1.16045e7;
Real dlogkT,T_max_tab,T_min_tab;
Real X = 1e-15; //0.7;   // Hydrogen Fraction
//Real Z_sun = 0.02;  //Metalicity
Real muH = 1./X;
Real mue = 2./(1. + X);


//Lodders et al 2003
Real Z_o_X_sun = 0.0177;
Real X_sun = 0.7491;
Real Y_sun =0.2246 + 0.7409 * (Z_o_X_sun);
Real Z_sun = 1.0 - X_sun - Y_sun;
Real muH_sun = 1./X_sun;

Real r_inner_boundary;


#define CUADRA_COOL (0)
#if (CUADRA_COOL==1)
Real Z = 3.*Z_sun;
#else
Real Z = 3.*Z_sun;
#endif
#if (CUADRA_COOL==1)
Real mu_highT = 0.5;
#else
Real mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.);  //mean molecular weight in proton masses
#endif

 void cons_force(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
 void emf_source(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim,  const AthenaArray<Real> &bcc, const AthenaArray<Real> &cons, EdgeField &e);
 void star_update_function(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
 void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
 void Dirichlet_Boundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
 void DirichletInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void interp_inits(const Real x, const Real y, const Real z, Real *rho, Real *vx, Real *vy, Real *vz, Real *p);

int RefinementCondition(MeshBlock *pmb);
Real DivergenceB(MeshBlock *pmb, int iout);


/* A structure defining the properties of each of the source 'stars' */
typedef struct Stars_s{
  Real M;     /* mass of the star */
  Real Mdot;    /* mass loss rate from solar wind (in M_solar/kyr) */
  Real Vwind;   /* speed of solar wind (in pc/kyr) */
  Real x1;      /* position in X,Y,Z (in pc) */
  Real x2;
  Real x3;
  int i;      /* i,j,k of x,y,z cell the star is located in */
  int j;
  int k;
  Real v1;      /* velocity in X,Y,Z */
  Real v2;
  Real v3;
  Real alpha;     /* euler angles for ZXZ rotation*/
  Real beta;
  Real gamma;
  Real tau;
  Real mean_angular_motion;
  Real eccentricity;
  Real rotation_matrix[3][3];
  Real period;
  Real radius;  /* effective radius of star */
  Real volume;   /* effective volume of star */
  Real B_A;     /* B at Alfven radus of star */
  Real r_A;     /* Alfven radius of star */
  RegionSize block_size;   /* block size of the mesh block in which the star is located */
  AthenaArray<Real> spin_axis;      /* angular momentum unit vector for star */
  AthenaArray<Real> x_axis;      /* x_axis for star frame */
  AthenaArray<Real> y_axis;      /* y_axis for star frame */
}Stars;


/* Initialize a couple of the key variables used throughout */
//Real r_inner_boundary = 0.;         /* remove mass inside this radius */
Real r_min_inits = 1e15; /* inner radius of the grid of initial conditions */
Stars star[500];          /* The stars structure used throughout */
int nstars;             /* Number of stars in the simulation */
static Real G = 4.48e-9;      /* Gravitational constant G (in problem units) */
Real gm_;               /* G*M for point mass at origin */
Real gm1;               /* \gamma-1 (adiabatic index) */
Real N_cells_per_radius;  /* Number of cells that are contained in one stellar radius (this is defined in terms of the longest length 
across a cell ) */
double SMALL = 1e-20;       /* Small number for numerical purposes */
LogicalLocation *loc_list;              /* List of logical locations of meshblocks */
int n_mb = 0; /* Number of meshblocks */
int max_smr_level = 0;
int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
Real beta_star;  /* beta for each star, defined wrt rho v^2 */

int N_r =128;  /* Number of points to sample in r for radiale profile */
int N_user_vars = 27; /* Number of user defined variables in UserWorkAfterLoop */
int N_user_history_vars = 26; /* Number of user defined variables which have radial profiles */
int N_user_vars_field = 6; /* Number of user defined variables related to magnetic fields */
Real r_dump_min,r_dump_max; /* Range in r to sample for radial profile */


Real yr = 31556926.0, pc = 3.09e18;    /* yr and parsec in code units */
Real cl = 2.99792458e10 * (1e3 * yr)/pc ;      /* speed of light in code units */
Real cs_max = cl ; //0.023337031 * cl;  /*sqrt(me/mp) cl....i.e. sound speed of electrons is ~ c */

bool amr_increase_resolution; /* True if resolution is to be increased from restarted run */

// int nx_inits,ny_inits,nz_inits; /* size of initial condition arrays */
// AthenaArray<Real> x_inits,y_inits,z_inits,v1_inits,v2_inits,v3_inits,press_inits,rho_inits; /* initial condition arrays*/


#if (CUADRA_COOL==0)
Real kbTfloor_kev = 8.00000000e-04;
#else
Real kbTfloor_kev = 8.61733130e-4;
#endif

/* global definitions for the cooling curve using the
   Townsend (2009) exact integration scheme */
//#define nfit_cool 11

#define nfit_cool 13
//Real kbT_cc[nfit_cool],Lam_cc[nfit_cool],
Real exp_cc[nfit_cool],Lam_cc[nfit_cool];

// #if (CUADRA_COOL ==0)
// static const Real kbT_cc[nfit_cool] = {
//   8.61733130e-06,   8.00000000e-04,   1.50000000e-03,
//   2.50000000e-03,   7.50000000e-03,   2.00000000e-02,
//   3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
//   2.26000000e+00,   1.00000000e+02};

// static const Real Lam_cc[nfit_cool] = {
//   1.24666909e-27,   3.99910139e-26,   1.47470970e-22,
//   1.09120314e-22,   4.92195285e-22,   5.38853593e-22,
//   2.32144473e-22,   1.38278507e-22,   3.66863203e-23,
//   2.15641313e-23,   9.73848346e-23};

// static const Real exp_cc[nfit_cool] = {
//    0.76546122,  13.06493514,  -0.58959508,   1.37120661,
//    0.09233853,  -1.92144798,  -0.37157016,  -1.51560627,
//   -0.26314206,   0.39781441,   0.39781441};
// #else
// static const Real kbT_cc[nfit_cool] = {
//     8.61733130e-06,   8.00000000e-04,   1.50000000e-03,
//     2.50000000e-03,   7.50000000e-03,   2.00000000e-02,
//     3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
//     2.26000000e+00,   1.00000000e+02};

// static const Real Lam_cc[nfit_cool] = {
//     1.89736611e-19,   7.95699530e-21,   5.12446122e-21,
//     3.58388517e-21,   1.66099838e-21,   8.35970776e-22,
//     6.15118667e-22,   2.31779533e-22,   1.25581948e-22,
//     3.05517566e-23,   2.15234602e-24};

// static const Real exp_cc[nfit_cool] = {
//     -0.7,  -0.7,  -0.7,   -0.7,
//     -0.7,  -0.7,  -0.7,  -0.7,
//     -0.7,   -0.7,  -0.7};
// #endif

static const Real kbT_cc[nfit_cool] = {
    8.00000000e-04,   1.50000000e-03, 2.50000000e-03,
    7.50000000e-03,   2.00000000e-02, 3.10000000e-02,
    1.25000000e-01,   3.00000000e-01, 8.22000000e-01,
    2.26000000e+00, 3.010000000e+00,  3.4700000000e+01,
    1.00000000e+02};
static const Real Lam_cc_H[nfit_cool] = {
    6.16069200e-24,   4.82675600e-22,   1.17988800e-22,
    1.08974000e-23,   3.59794100e-24,   2.86297800e-24,
    2.85065300e-24,   3.73480000e-24,   5.58385400e-24,
    8.75574600e-24,   1.00022900e-23,   3.28378800e-23,
    5.49397977e-23};
static const Real Lam_cc_He[nfit_cool] = {
    6.21673000e-29,   5.04222000e-26,   4.57500800e-24,
     1.06434700e-22,   1.27271300e-23,   6.06418700e-24,
     1.81856400e-24,   1.68631400e-24,   2.10823800e-24,
     3.05093700e-24,   3.43240700e-24,   1.02736900e-23,
     1.65141824e-23};
static const Real Lam_cc_Metals[nfit_cool] = {
    5.52792700e-26,   5.21197100e-24,   1.15916400e-22,
     1.17551800e-21,   1.06054500e-21,   2.99407900e-22,
     1.04016100e-22,   2.40264000e-23,   2.14343200e-23,
     5.10608800e-24,   4.29634900e-24,   4.12127500e-24,
     4.04771017e-24 };

static Real Yk[nfit_cool];
/* -- end piecewise power-law fit */


/* must call init_cooling() in both problem() and read_restart() */
static void init_cooling();
static void init_cooling_tabs(std::string filename);
static void test_cooling();
static Real newtemp_townsend(const Real d, const Real T, const Real dt_hydro);



static void init_cooling_tabs(std::string filename){
//  FILE *input_file;
//    if ((input_file = fopen(filename.c_str(), "r")) == NULL)  { 
//           fprintf(stderr, "Cannot open %s, %s\n", "input_file",filename.c_str());
//           exit(0);
//         }

  Real T_Kelvin, tmp, Lam_H, Lam_metals ; 
  int j;
  for (j=0; j<nfit_cool; j++) {
//   fscanf(input_file, "%lf %lf %lf \n",&T_Kelvin,&Lam_H,&Lam_metals);

//   kbT_cc[j] = T_Kelvin /keV_to_Kelvin;

//Lam_cc[j] = Lam_H + Lam_metals * Z/Z_sun;
   Lam_cc[j] = X/X_sun * Lam_cc_H[j] + (1.-X-Z)/Y_sun * Lam_cc_He[j] + Z/Z_sun * Lam_cc_Metals[j];
 }

 //fclose(input_file);
 for (j=0; j<nfit_cool-1; j++) exp_cc[j] = std::log(Lam_cc[j+1]/Lam_cc[j])/std::log(kbT_cc[j+1]/kbT_cc[j]) ; 

  exp_cc[nfit_cool-1] = exp_cc[nfit_cool -2];
    
    T_min_tab = kbT_cc[0];
    T_max_tab = kbT_cc[nfit_cool-1];
    dlogkT = std::log(kbT_cc[1]/kbT_cc[0]);


//for (j=0; j<nfit_cool; j++) fprintf(stderr,"kbT: %g Lam: %g exp_cc: %g \n", kbT_cc[j], Lam_cc[j],exp_cc[j]);

   
   return ;
}


static void init_cooling()
{
  int k, n=nfit_cool-1;
  Real term;

  /* populate Yk following equation A6 in Townsend (2009) */
  Yk[n] = 0.0;
  for (k=n-1; k>=0; k--){
    term = (Lam_cc[n]/Lam_cc[k]) * (kbT_cc[k]/kbT_cc[n]);

    if (exp_cc[k] == 1.0)
      term *= log(kbT_cc[k]/kbT_cc[k+1]);
    else
      term *= ((1.0 - std::pow(kbT_cc[k]/kbT_cc[k+1], exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

    Yk[k] = Yk[k+1] - term;
  }
  return;
}

/* piecewise power-law fit to the cooling curve with temperature in
   keV and L in erg cm^3 / s */
static Real Lambda_T(const Real T)
{
  int k, n=nfit_cool-1;

  /* first find the temperature bin */
  for(k=n; k>=1; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* piecewise power-law; see equation A4 of Townsend (2009) */
  /* (factor of 1.311e-5 takes lambda from units of 1e-23 erg cm^3 /s
     to code units.) */
  return (Lam_cc[k] * std::pow(T/kbT_cc[k], exp_cc[k]));
}

/* see Lambda_T() or equation A1 of Townsend (2009) for the
   definition */
static Real Y(const Real T)
{
  int k, n=nfit_cool-1;
  Real term;

  /* first find the temperature bin */
  for(k=n; k>=1; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* calculate Y using equation A5 in Townsend (2009) */
  term = (Lam_cc[n]/Lam_cc[k]) * (kbT_cc[k]/kbT_cc[n]);

  if (exp_cc[k] == 1.0)
    term *= log(kbT_cc[k]/T);
  else
    term *= ((1.0 - std::pow(kbT_cc[k]/T, exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

  return (Yk[k] + term);
}

static Real Yinv(const Real Y1)
{
  int k, n=nfit_cool-1;
  Real term;

  /* find the bin i in which the final temperature will be */
  for(k=n; k>=1; k--){
    if (Y(kbT_cc[k]) >= Y1)
      break;
  }


  /* calculate Yinv using equation A7 in Townsend (2009) */
  term = (Lam_cc[k]/Lam_cc[n]) * (kbT_cc[n]/kbT_cc[k]);
  term *= (Y1 - Yk[k]);

  if (exp_cc[k] == 1.0)
    term = exp(-1.0*term);
  else{
    term = std::pow(1.0 - (1.0-exp_cc[k])*term,
               1.0/(1.0-exp_cc[k]));
  }

  return (kbT_cc[k] * term);
}

static Real newtemp_townsend(const Real d, const Real T, const Real dt_hydro)
{
  Real term1, Tref;
  int n=nfit_cool-1;

  Tref = kbT_cc[n];

  term1 = (T/Tref) * (Lambda_T(Tref)/Lambda_T(T)) * (dt_hydro/tcool(d, T));

  return Yinv(Y(T) + term1);
}

static Real tcool(const Real d, const Real T)
{
  // T is in keV, d is in g/cm^3
#if (CUADRA_COOL==0)
  //return  (T) * (muH * muH) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
    return  (T) * (muH_sun * mue) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
#else
  return  (T) * (mu_highT) / ( gm1 * d *             Lambda_T(T)/UnitLambda_times_mp_times_kev );
#endif
}


static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k, dk;
  int is, ie, js, je, ks, ke;


  ///apply_inner_boundary_condition(pmb,prim);

  Real kbT_keV;
  AthenaArray<Real> prim_before,cons;


    // Allocate memory for primitive/conserved variables
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (pmb->block_size.nx2 > 1) ncells2 = pmb->block_size.nx2 + 2*(NGHOST);
  if (pmb->block_size.nx3 > 1) ncells3 = pmb->block_size.nx3 + 2*(NGHOST);
  prim_before.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
  cons.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
  prim_before = prim;
  // //prim.InitWithShallowCopy(pmb->phydro->w);

  // /* ath_pout(0, "integrating cooling using Townsend (2009) algorithm.\n"); */

  is = pmb->is;  ie = pmb->ie;
  js = pmb->js;  je = pmb->je;
  ks = pmb->ks;  ke = pmb->ke;
  Real igm1 = 1.0/(gm1);
  Real gamma = gm1+1.;

  //pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, pmb->pfield->bcc,
           //pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);


  
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {

        /* find temp in keV */
        // kbT_keV = mu_highT*mp_over_kev*(prim(IPR,k,j,i)/prim(IDN,k,j,i));
        // // ath_pout(0, "temperature before = %e ", kbT_keV);
        // kbT_keV = newtemp_townsend(prim(IDN,k,j,i), kbT_keV, dt_hydro);
        // // ath_pout(0, "temperature after = %e \n", kbT_keV);
        // // apply a temperature floor (nans tolerated) 
        // if (isnan(kbT_keV) || kbT_keV < kbTfloor_kev)
        //   kbT_keV = kbTfloor_kev;

        // prim(IPR,k,j,i) = prim(IDN,k,j,i) * kbT_keV / (mu_highT * mp_over_kev);

        Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

        if (v_s>cs_max) v_s = cs_max;
        if ( fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
        if ( fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
        if ( fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

         prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;
       // cons(IEN,k,j,i) = prim(IPR,k,j,i)*igm1 + 0.5*prim(IDN,k,j,i)*( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
          
      }
    }
  }




  apply_inner_boundary_condition(pmb,prim);



      prim_before.DeleteAthenaArray();
      cons.DeleteAthenaArray();
  return;
}



/******************************************/
/*        Some Vector Functions           */
/******************************************/


void cross(const AthenaArray<Real> &A , const AthenaArray<Real> &B, AthenaArray<Real> &result){
    
    result(0) = A(1)*B(2) - A(2)*B(1);
    result(1) = A(2)*B(0) - A(0)*B(2);
    result(2) = A(0)*B(1) - A(1)*B(0);
    return;
    
}

Real dot(const AthenaArray<Real> &A , const AthenaArray<Real> &B){
    return A(0) * B(0) + A(1) * B(1) + A(2) * B(2);
}

Real norm_calc(const AthenaArray<Real> &A ){
    return std::sqrt( SQR(A(0)) + SQR(A(1)) + SQR(A(2)) );
}
void norm_vector(AthenaArray<Real> &A){
    Real norm = norm_calc(A);
    for (int i=0; i<=2; ++i) A(i) *= 1./norm;
    return;
}

void add_vectors(const int i_sign, const AthenaArray<Real> &A , const AthenaArray<Real> &B, AthenaArray<Real> &result){
  for (int i=0; i<=2; ++i) result(i) = A(i) + i_sign* B(i);
  return;
}
void scale_vector(const AthenaArray<Real> &A , const Real alpha , AthenaArray<Real> &result){
  for (int i=0; i<=2; ++i) result(i) = A(i)*alpha;
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

Real get_dV(const Coordinates *pcoord, const Real x1, const Real x2, const Real x3,const Real dx1, const Real dx2, const Real dx3){
    
    if (COORDINATE_SYSTEM == "cartesian"){
        if (pcoord->pmy_block->block_size.nx3>1){
            return dx1 * dx2 * dx3;
        }
        else{
            return dx1 * dx2;
        }
    }
    else if (COORDINATE_SYSTEM == "cylindrical"){
        
        if (pcoord->pmy_block->block_size.nx3>1){
            return dx1 * x1 * dx2 * dx3;
        }
        else{
            return dx1 * x1 * dx2 ;
        }
        
    }
    else if (COORDINATE_SYSTEM == "spherical_polar"){
        
        return dx1 * x1 * dx2 * x1 * std::sin(x2) *dx3 ;
    }
    
}

/*
Convert vector in cartesian coords to code coords
*/
void convert_cartesian_vector_to_code_coords(const Real vx, const Real vy, const Real vz, const Real x, const Real y, const Real z, Real *vx1, Real *vx2, Real *vx3){

  if (COORDINATE_SYSTEM == "cartesian"){
    *vx1 = vx;
    *vx2 = vy;
    *vx3 = vz;
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){

    Real s = sqrt( SQR(x) + SQR(y) );

    *vx1 = vx * x/s + vy * y/s;
    *vx2 =(-y * vx + x * vy) / (s);
    *vx3 = vz;

  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    Real r = sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real s = sqrt( SQR(x) + SQR(y) );


    *vx1 = vx * x/r + vy * y/r + vz *z/r;
    *vx2 = ( (x * vx + y * vy) * z  - SQR(s) * vz ) / (r * s + SMALL) ;
    *vx3 = (-y * vx + x * vy) / (s + SMALL) ;
  }

}
/*
Returns approximate cell sizes if the grid was uniform
*/

void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ){

  if (COORDINATE_SYSTEM == "cartesian"){
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



int is_in_block(RegionSize block_size, const Real x, const Real y, const Real z){
  Real x1, x2, x3;

  if (COORDINATE_SYSTEM =="cartesian"){
    x1 = x;
    x2 = y;
    x3 = z;
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    x1 = sqrt(x*x + y*y);
    x2 = std::atan2(y,x);
    x3 = z;
  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    x1 = sqrt(x*x + y*y + z*z);
    x2 = std::acos(z/x1);
    x3 = std::atan2(y,x);
  }

  int is_in_x1 = (block_size.x1min <= x1) && (block_size.x1max >= x1);
  int is_in_x2 = (block_size.x2min <= x2) && (block_size.x2max >= x2);
  int is_in_x3 = (block_size.x3min <= x3) && (block_size.x3max >= x3);

  if (block_size.nx3>1) return is_in_x3 * is_in_x2 * is_in_x1;
  else return is_in_x2 * is_in_x1;


}

void get_minimum_cell_lengths(const Coordinates * pcoord, Real *dx_min, Real *dy_min, Real *dz_min){
    
        //loc_list = pcoord->pmy_block->pmy_mesh->loclist; 
        RegionSize block_size;
        enum BoundaryFlag block_bcs[6];
        //int n_mb = pcoord->pmy_block->pmy_mesh->nbtotal;
    
        *dx_min = 1e15;
        *dy_min = 1e15;
        *dz_min = 1e15;

	Real DX,DY,DZ; 
        if (amr_increase_resolution){
	   get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ); 
	
	*dx_min = DX/std::pow(2.,max_refinement_level);
	*dy_min = DY/std::pow(2.,max_refinement_level);
	*dz_min = DZ/std::pow(2.,max_refinement_level);
	return;
	} 
        
        block_size = pcoord->pmy_block->block_size;
    
        
        /* Loop over all mesh blocks by reconstructing the block sizes to find the block that the
         star is located in */
        for (int j=0; j<n_mb; j++) {
            pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loc_list[j], block_size, block_bcs);
            
            get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
            
            if (DX < *dx_min) *dx_min = DX;
            if (DY < *dy_min) *dy_min = DY;
            if (DZ < *dz_min) *dz_min = DZ;
            
            
            
        }
        
    
}
/* Make sure i,j,k are in the domain */
void bound_ijk(Coordinates *pcoord, int *i, int *j, int*k){
    
    int is,js,ks,ie,je,ke;
    is = pcoord->pmy_block->is;
    js = pcoord->pmy_block->js;
    ks = pcoord->pmy_block->ks;
    ie = pcoord->pmy_block->ie;
    je = pcoord->pmy_block->je;
    ke = pcoord->pmy_block->ke;
    
    
    if (*i<is) *i = is;
    if (*j<js) *j = js;
    if (*k<ks) *k = ks;
    
    if (*i>ie) *i = ie;
    if (*j>je) *j = je;
    if (*k>ke) *k = ke;
    
    return; 
}


/* Set maximum temperature such that v_th <= c */
void limit_temperature(MeshBlock *pmb,AthenaArray<Real> &cons,const AthenaArray<Real> &prim_old){

  AthenaArray<Real> prim;
  prim.InitWithShallowCopy(pmb->phydro->w);
  pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, pmb->pfield->bcc,
           pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
    Real gamma = gm1 +1.;

for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

    Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

        if (v_s>cs_max) v_s = cs_max;
          
        if ( fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
        if ( fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
        if ( fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

    prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;



  }}}


  pmb->peos->PrimitiveToConserved(prim, pmb->pfield->bcc,
       cons, pmb->pcoord,
       pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

  prim.DeleteAthenaArray();

  return;

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

            Real xprime = x-20.0;
            Real yprime = y;
            Real zprime = z;

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

/* Apply inner "inflow" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


  Real v_ff = std::sqrt(2.*gm_/(pmb->r_inner_boundary+SMALL));
  Real va_max; /* Maximum Alfven speed allowed */
  Real bsq,bsq_rho_ceiling;

  Real dx,dy,dz,dx_min,dy_min,dz_min;
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);  /* spacing of this block */

  get_minimum_cell_lengths(pmb->pcoord, &dx_min, &dy_min, &dz_min); /* spacing of the smallest block */

  /* Allow for larger Alfven speed if grid is coarser */
  va_max = v_ff * std::sqrt(dx/dx_min);


  Real r,x,y,z;
   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


          get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);
          Real r1,r2,new_x,new_y,new_z, r_hat_x,r_hat_y,r_hat_z;
          Real dE_dr, drho_dr,dM1_dr,dM2_dr,dM3_dr;
          Real dU_dr;
          int is_3D,is_2D;
          int i1,j1,k1,i2,j2,k2,ir;
          Real m_r;
          
          is_3D = (pmb->block_size.nx3>1);
          is_2D = (pmb->block_size.nx2>1);

          r = sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);

          if (MAGNETIC_FIELDS_ENABLED){

            bsq = SQR(pmb->pfield->bcc(IB1,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)) + SQR(pmb->pfield->bcc(IB3,k,j,i));
            bsq_rho_ceiling = SQR(va_max);

          

            if (prim(IDN,k,j,i) < bsq/bsq_rho_ceiling){
              pmb->user_out_var(16,k,j,i) += bsq/bsq_rho_ceiling - prim(IDN,k,j,i);
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }

            }

          if (r < pmb->r_inner_boundary){
              
              // r_hat_x = x/r;
              // r_hat_y = y/r;
              // r_hat_z = z/r;
              // r1 = r;
              // ir = 1;


              // while (r1 <r_inner_boundary) {
                  
              //      i1 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              //      j1 = int (j + (ir*y/r + 0.5)*is_2D);
              //      k1 = int (k + (ir*z/r + 0.5)*is_3D);
                  
              //      bound_ijk(pmb->pcoord,&i1,&j1,&k1);
                  
              //      r1 = sqrt( SQR(pmb->pcoord->x1v(i1)) + SQR(pmb->pcoord->x2v(j1))*is_2D + SQR(pmb->pcoord->x3v(k1))*is_3D );
              //      ir = ir + 1;
              //  }

              // while (r1 <r_inner_boundary) {
                  
              //     i1 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              //     j1 = int (j + (ir*y/r + 0.5)*is_2D);
              //     k1 = int (k + (ir*z/r + 0.5)*is_3D);
                  
              //     bound_ijk(pmb->pcoord,&i1,&j1,&k1);
                  
              //     r1 = sqrt( SQR(pmb->pcoord->x1v(i1)) + SQR(pmb->pcoord->x2v(j1))*is_2D + SQR(pmb->pcoord->x3v(k1))*is_3D );
              //     ir = ir + 1;
              // }

              // //fprintf(stderr,"r: %g r_inner: %g r1: %g ijk: %d %d %d i1j1k1: %d %d %d \n", r, r_inner_boundary, r1,i,j,k,i1,j1,k1);

              
              // i2 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              // j2 = int (j + (ir*y/r + 0.5)*is_2D);
              // k2 = int (k + (ir*z/r + 0.5)*is_3D);
              
              // bound_ijk(pmb->pcoord,&i2,&j2,&k2);
              // r2 = sqrt( SQR(pmb->pcoord->x1v(i2)) + SQR(pmb->pcoord->x2v(j2))*is_2D + SQR(pmb->pcoord->x3v(k2))*is_3D );
              

              // for (int n=0; n<((NHYDRO+NFIELD)); ++n) {
                  
                  
              //     dU_dr =(cons(n,k2,j2,i2) - cons(n,k1,j1,i1)) /  (r2-r1 + SMALL);
                  
              //     cons(n,k,j,i) = cons(n,k1,j1,i1); //+ dU_dr * (r-r1);
                  
              // }
              
              Real rho_flr = 1e-7;
              Real p_floor = 1e-10;
              if (MAGNETIC_FIELDS_ENABLED){

                bsq_rho_ceiling = SQR(va_max);
                Real new_rho = bsq/bsq_rho_ceiling;


                if (new_rho>rho_flr) rho_flr = new_rho;

            }

              prim(IDN,k,j,i) = rho_flr;
              prim(IVX,k,j,i) = 0.;
              prim(IVY,k,j,i) = 0.;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = p_floor;
            
              Real drho = prim(IDN,k,j,i) - rho_flr;
              pmb->user_out_var(N_user_history_vars,k,j,i) += drho;
              // cons(IDN,k,j,i) = rho_flr;
              // cons(IM1,k,j,i) = 0.;
              // cons(IM2,k,j,i) = 0.;
              // cons(IM3,k,j,i) = 0.;
              // cons(IEN,k,j,i) = p_floor/gm1;

              // for (int n=0; n<((NHYDRO+NFIELD)); ++n) {
                 
              //     cons(n,k,j,i) = cons(n,k1,j1,i1); 
                 
              // }

              /* Prevent outflow from inner boundary */ 
              if (prim(IVX,k,j,i)*x/r >0 ) prim(IVX,k,j,i) = 0.;
              if (prim(IVY,k,j,i)*y/r >0 ) prim(IVY,k,j,i) = 0.;
              if (prim(IVZ,k,j,i)*z/r >0 ) prim(IVZ,k,j,i) = 0.;

              
              
          }



}}}



}


/* Convert position to location of cell.  Note well: assumes a uniform grid in each meshblock */
void get_ijk(MeshBlock *pmb,const Real x, const Real y, const Real z , int *i, int *j, int *k){
    Real dx = pmb->pcoord->dx1f(0);
    Real dy = pmb->pcoord->dx2f(0);
    Real dz = pmb->pcoord->dx3f(0);

   *i = int ( (x-pmb->block_size.x1min)/dx) + pmb->is;
   *j = int ( (y-pmb->block_size.x2min)/dy) + pmb->js;
   *k = pmb->ks;

   if (*i>pmb->ie) *i = pmb->ie;
   if (*j>pmb->je) *j = pmb->je;

   if (pmb->block_size.nx3>1) *k = int ( (z-pmb->block_size.x3min)/dz) + pmb->ks;

   if (*k>pmb->ke) *k = pmb->ke;
   // if ( (x < pmb->pcoord->x1f(*i) ) || (x > pmb->pcoord->x1f(*i+1) ) ||
   //      (y < pmb->pcoord->x2f(*j) ) || (y > pmb->pcoord->x2f(*j+1) ) ){
   //        fprintf(stderr,"Error in get_ijk, ijk : %d %d %d outside of cell for xyz: %g %g %g \n",*i,*j,*k,x,y,z);
   //        fprintf(stderr,"Error in get_ijk, x1_bound : %g %g \n x2_bound : %g %g\n",pmb->pcoord->x1f(*i),pmb->pcoord->x1f(*i+1),pmb->pcoord->x2f(*j),pmb->pcoord->x2f(*j+1));

   //        exit(0);
   //      }

   //  if (pmb->block_size.nx3>1){
   //    if ( (z < pmb->pcoord->x3f(*k) ) || (z > pmb->pcoord->x3f(*k+1) ) ){
   //      fprintf(stderr,"Error in get_ijk, ijk : %d %d %d outside of cell for xyz: %g %g %g",*i,*j,*k,x,y,z);
   //        exit(0);
   //    }
   //  } 
    
}

/* Compute the total mass removed from the inner boundary.  Cumulative over the whole simulation */
Real compute_mass_removed(MeshBlock *pmb, int iout){
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  AthenaArray<Real> vol;
  vol.NewAthenaArray(ncells1);

  Real sum = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
        sum += pmb->user_out_var(N_user_history_vars,k,j,i) * vol(i);
      }
    }
  }

  vol.DeleteAthenaArray();

  return sum;
}

/* Compute the total mass enclosed in the inner boundary.   */
Real compute_mass_in_boundary(MeshBlock *pmb, int iout){
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  AthenaArray<Real> vol;
  vol.NewAthenaArray(ncells1);

  Real sum = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
        Real x,y,z;
        
        int is_3D = (pmb->block_size.nx3>1);
        int is_2D = (pmb->block_size.nx2>1);
        
        get_cartesian_coords(pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k),&x,&y,&z);
        
        Real r = std::sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);


        if (r<pmb->r_inner_boundary) sum += pmb->phydro->u(IDN,k,j,i) * vol(i);
      }
    }
  }

  vol.DeleteAthenaArray();

  return sum;
}

/* Compute the average of primitives and user_out_vars over angle   */
Real radial_profile(MeshBlock *pmb, int iout){
    int i_r = iout % N_r;
    int i_prim =  iout / N_r + IDN;
    
    int i_user_var = i_prim - (NHYDRO) - IDN -1;


    r_dump_min = pmb->r_inner_boundary/2.;
    r_dump_max = pmb->pmy_mesh->mesh_size.x1max;
    
    Real r = r_dump_min * std::pow(10., (i_r * std::log10(r_dump_max/r_dump_min)/(1.*N_r-1.)) );


    if (i_prim == (NHYDRO)) return r/(1.*pmb->pmy_mesh->nbtotal);

    
    int N_phi = 64 ;
    int N_theta = 1;
    
    if (pmb->block_size.nx3>1) N_theta = 64;
    
    Real dphi = 2.*PI/(N_phi*1.-1.);
    
    Real dtheta = 1.;
    Real Omega = 2.*PI;

    if (pmb->block_size.nx3>1) {
      dtheta = PI/(N_theta*1.-1.);
      Omega = 4.*PI;
    }
    

    Real result = 0.;
        for (int i_theta=0; i_theta<N_theta; ++i_theta) {
            for (int i_phi=0; i_phi<N_phi; ++i_phi) {
                Real phi = i_phi * dphi;
                Real theta = PI/2.;
                int i,j,k;
                
                if (pmb->block_size.nx3>1) theta = i_theta * dtheta;
                
                Real x = r * std::cos(phi) * std::sin(theta);
                Real y = r * std::sin(phi) * std::sin(theta);
                Real z = r * std::cos(theta);
                Real fac = 1.;
                if ( is_in_block(pmb->block_size,x,y,z) ){
                    
                    get_ijk(pmb,x,y,z,&i,&j,&k);
                    

                    if (i_phi == 0 || i_phi == N_phi-1) fac = fac*0.5;
                    if ( (i_theta ==0 || i_theta == N_theta-1) && (pmb->block_size.nx3>1) ) fac = fac*0.5;

                    Real dOmega =  std::sin(theta)*dtheta*dphi * fac;
                    if (i_prim<=IPR){
                        result += pmb->phydro->w(i_prim,k,j,i) * dOmega / Omega;
                    }
                    else{
                        result += pmb->user_out_var(i_user_var,k,j,i) * dOmega / Omega;
                        
                    }
                    
                    
                }
                
                
                
            }
        }
    
    return result;
    
    
}

// /* Interpolate fluid inital conditions to computational grid */
// void interp_inits(const Real x, const Real y, const Real z, Real *rho, Real *vx, Real *vy, Real *vz, Real *p){


//  Real dx = x_inits(0,0,1) - x_inits(0,0,0);
//  Real dy = y_inits(0,1,0) - y_inits(0,0,0);
//  Real dz = z_inits(1,0,0) - z_inits(0,0,0);

//  Real x0 = x_inits(0,0,0);
//  Real y0 = y_inits(0,0,0);
//  Real z0 = z_inits(0,0,0);

//  int i = (int) ((x - x0) / dx + 0.5 + 1000) - 1000;
//  int j = (int) ((y - y0) / dy + 0.5 + 1000) - 1000;
//  int k = (int) ((z - z0) / dz + 0.5 + 1000) - 1000;
    

    
//     //fprintf(stderr,"x y z: %g %g %g \n dx dy dz: %g %g %g \n x0 y0 z0: %g %g %g \n i j k: %d %d %d \n",x,y,z,dx,dy,dz,x0,y0,z0,i,j,k);
//    //fprintf(stderr,"nx ny nz: %d %d %d\n", nx_inits,ny_inits,nz_inits);

//  //fprintf(stderr,"x y z: %g %g %g \n x_inits y_inits z_inits: %g %g %g \n", x,y,z, x_inits(k,j,i),y_inits(k,j,i),z_inits(k,j,i));

//  Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
//  if (i<0 || i>=nx_inits || j<0 || j>=ny_inits || k<0 || k>=nz_inits || r<r_min_inits){
//   *rho = 1e-8;
//   *vx = 0.;
//   *vy = 0.;
//   *vz = 0.;
//   *p = 1e-10;
//  }
//  else{
//   *rho = rho_inits(k,j,i);
//   *vx = v1_inits(k,j,i);
//   *vy = v2_inits(k,j,i);
//   *vz = v3_inits(k,j,i);
//   *p = press_inits(k,j,i);

//  }


// }

/*
 * -------------------------------------------------------------------
 *     Initialize Mesh
 * -------------------------------------------------------------------
 */
void Mesh::InitUserMeshData(ParameterInput *pin)
{



    
    if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


    // EnrollUserRadSourceFunction(integrate_cool);
    
    int i = 0;
    if (MAGNETIC_FIELDS_ENABLED){
        N_user_history_vars += N_user_vars_field;
        N_user_vars += N_user_vars_field;
    }

    // loc_list = loclist;
    // n_mb = nbtotal;
    // AllocateUserHistoryOutput(33*3);
    // for (int i = 0; i<33*3; i++){
    //     int i_star = i/3;
    //     int i_pos  = i % 3;
    //     EnrollUserHistoryOutput(i, star_position, "star_"); 
    // }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){

    
    AllocateUserOutputVariables(N_user_vars);

    r_inner_boundary = 0.;
    loc_list = pmy_mesh->loclist;
    n_mb = pmy_mesh->nbtotal;
    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    Real horizon_radius = 2.0 * gm_/SQR(cl);

    gm1 = peos->GetGamma() - 1.0;


    std::string file_name,cooling_file_name;
    cooling_file_name = pin->GetOrAddString("problem","cooling_file","lambda.tab");
    int N_cells_per_boundary_radius = pin->GetOrAddInteger("problem", "boundary_radius", 2);


    max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);
    max_smr_level = pin->GetOrAddReal("mesh","smrlevel",max_refinement_level); 
    if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;
    
    Real dx_min,dy_min,dz_min;
    get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);
    
    if (block_size.nx3>1)       r_inner_boundary = N_cells_per_boundary_radius * std::max(std::max(dx_min,dy_min),dz_min); // r_inner_boundary = 2*sqrt( SQR(dx_min) + SQR(dy_min) + SQR(dz_min) );
    else if (block_size.nx2>1)  r_inner_boundary = N_cells_per_boundary_radius * std::max(dx_min,dy_min); //2*sqrt( SQR(dx_min) + SQR(dy_min)               );
    else                        r_inner_boundary = N_cells_per_boundary_radius * dx_min;

    if (r_inner_boundary < horizon_radius) r_inner_boundary = horizon_radius;

    Real v_ff = std::sqrt(2.*gm_/(r_inner_boundary+SMALL))*10.;
    cs_max = std::min(cs_max,v_ff);
    
    
    init_cooling_tabs(cooling_file_name);
    init_cooling();

    
}


/* 
* -------------------------------------------------------------------
*     The actual problem / initial condition setup file
* -------------------------------------------------------------------
*/
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int i=0,j=0,k=0;

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
  Real pressure,b0,da,pa,ua,va,wa,bxa,bya,bza,x1,x2;
  Real T_dt,T_dmin,T_dmax;



  Real gm1 = peos->GetGamma() - 1.0;
  /* Set up a uniform medium */
  /* For now, make the medium almost totally empty */
  da = 1.0;
  pa = 0.01;
  ua = 0.0;
  va = 0.0;
  wa = 0.0;
  bxa = 0.0;
  bya = 0.0;
  bza = 0.0;
  Real x,y,z;

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
  for (i=il; i<=iu; i++) {

        

      phydro->w(IDN,k,j,i) = da;
      phydro->w(IVX,k,j,i) = ua;
      phydro->w(IVY,k,j,i) = va;
      phydro->w(IVZ,k,j,i) = wa;
      phydro->w(IPR,k,j,i) = pa;

        

    phydro->u(IDN,k,j,i) = da;
    phydro->u(IM1,k,j,i) = da*ua;
    phydro->u(IM2,k,j,i) = da*va;
    phydro->u(IM3,k,j,i) = da*wa;

    for (int i_user=0;i_user<N_user_vars; i_user ++){
      user_out_var(i_user,k,j,i) = 0;
    }

if (MAGNETIC_FIELDS_ENABLED){
    pfield->b.x1f(k,j,i) = bxa;
    pfield->b.x2f(k,j,i) = bya;
    pfield->b.x3f(k,j,i) = bza;
    pfield->bcc(IB1,k,j,i) = bxa;
    pfield->bcc(IB2,k,j,i) = bya;
    pfield->bcc(IB3,k,j,i) = bza;
    if (i == ie) pfield->b.x1f(k,j,i+1) = bxa;
    if (j == je) pfield->b.x2f(k,j+1,i) = bya;
    if (k == ke) pfield->b.x3f(k+1,j,i) = bza;


        
}

    pressure = pa;
#ifndef ISOTHERMAL
    phydro->u(IEN,k,j,i) = pressure/gm1;
if (MAGNETIC_FIELDS_ENABLED){
      phydro->u(IEN,k,j,i) +=0.5*(bxa*bxa + bya*bya + bza*bza);
}
     phydro->u(IEN,k,j,i) += 0.5*da*(ua*ua + va*va + wa*wa);
#endif /* ISOTHERMAL */

      if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
        phydro->u(IEN,k,j,i) += da;
 
  }}}
    

    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord,
                                   il, iu, jl, ju, ks, ku);

    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord,
                                  il, iu, jl, ju, ks, ku);



  UserWorkInLoop();

  ///exit(0);


  

}



