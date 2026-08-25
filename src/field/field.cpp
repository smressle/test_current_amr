//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field.cpp
//! \brief implementation of functions in class Field

// C headers

// C++ headers
#include <string>
#include <vector>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "field.hpp"
#include "field_diffusion/field_diffusion.hpp"

// #include <cstdint>
// #include <cstring>

// bool IsNaN(double x) {
//     uint64_t u;
//     std::memcpy(&u, &x, sizeof(double));

//     uint64_t exp  = (u >> 52) & 0x7ffULL;
//     uint64_t frac = u & 0xfffffffffffffULL;

//     return (exp == 0x7ffULL) && (frac != 0);
// }
//! constructor, initializes data structures and parameters
// GCC/Clang


// #pragma GCC optimize("no-finite-math-only")
// bool check_nan(float x) {
//     return std::isnan(x);
// }

// bool isnan_volatile(float x) {
//     volatile float v = x;
//     return v != v;   // NaN is the only value not equal to itself
// }


Field::Field(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block(pmb), b(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    b1(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    bcc(NFIELD, pmb->ncells3, pmb->ncells2, pmb->ncells1),
    e(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    wght(pmb->ncells3, pmb->ncells2, pmb->ncells1),
    e2_x1f( pmb->ncells3   , pmb->ncells2   ,(pmb->ncells1+1)),
    e3_x1f( pmb->ncells3   , pmb->ncells2   ,(pmb->ncells1+1)),
    e1_x2f( pmb->ncells3   ,(pmb->ncells2+1), pmb->ncells1   ),
    e3_x2f( pmb->ncells3   ,(pmb->ncells2+1), pmb->ncells1   ),
    e1_x3f((pmb->ncells3+1), pmb->ncells2   , pmb->ncells1   ),
    e2_x3f((pmb->ncells3+1), pmb->ncells2   , pmb->ncells1   ),
    coarse_bcc_(3, pmb->ncc3, pmb->ncc2, pmb->ncc1,
                (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
                 AthenaArray<Real>::DataStatus::empty)),
    coarse_b_(pmb->ncc3, pmb->ncc2, pmb->ncc1+1,
              (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
               AthenaArray<Real>::DataStatus::empty)),
    fbvar(pmb, &b, coarse_b_, e),
    fdif(pmb, pin) {
  int ncells1 = pmb->ncells1, ncells2 = pmb->ncells2, ncells3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;

  pmb->RegisterMeshBlockData(b);

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  // Note the extra cell in each longitudinal direction for interface fields
  std::string integrator = pin->GetOrAddString("time","integrator","vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED) {
    // future extension may add "int nregister" to Hydro class
    b2.x1f.NewAthenaArray( ncells3   , ncells2   ,(ncells1+1));
    b2.x2f.NewAthenaArray( ncells3   ,(ncells2+1), ncells1   );
    b2.x3f.NewAthenaArray((ncells3+1), ncells2   , ncells1   );
  }

  if (STS_ENABLED) {
    std::string sts_integrator = pin->GetOrAddString("time", "sts_integrator", "rkl2");
    if (sts_integrator == "rkl2") {
      b0.x1f.NewAthenaArray( ncells3   , ncells2   ,(ncells1+1));
      b0.x2f.NewAthenaArray( ncells3   ,(ncells2+1), ncells1   );
      b0.x3f.NewAthenaArray((ncells3+1), ncells2   , ncells1   );
      ct_update.x1f.NewAthenaArray( ncells3   , ncells2   ,(ncells1+1));
      ct_update.x2f.NewAthenaArray( ncells3   ,(ncells2+1), ncells1   );
      ct_update.x3f.NewAthenaArray((ncells3+1), ncells2   , ncells1   );
    }
  }

  // Allocate memory for scratch vectors
  if (!pm->f3)
    cc_e_.NewAthenaArray(ncells3, ncells2, ncells1);
  else
    cc_e_.NewAthenaArray(3, ncells3, ncells2, ncells1);

  face_area_.NewAthenaArray(ncells1);
  edge_length_.NewAthenaArray(ncells1);
  edge_length_p1_.NewAthenaArray(ncells1);
  if (GENERAL_RELATIVITY) {
    g_.NewAthenaArray(NMETRIC, ncells1);
    gi_.NewAthenaArray(NMETRIC, ncells1);
  }

  if (pm->multilevel) {
    // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
    refinement_idx = pmy_block->pmr->AddToRefinement(&b, &coarse_b_);
  }

  // enroll FaceCenteredBoundaryVariable object
  fbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&fbvar);
  pmb->pbval->bvars_main_int.push_back(&fbvar);
  if (STS_ENABLED) {
    if (fdif.field_diffusion_defined) {
      if (!pmb->phydro->hdif.hydro_diffusion_defined && NON_BAROTROPIC_EOS) {
        pmb->pbval->bvars_sts.push_back(pmb->pbval->bvars_main_int[0]);
      }
      pmb->pbval->bvars_sts.push_back(&fbvar);
    }
  }



    emf_sourceterms_defined = false;
    UserEMFSourceTerm = pmy_block->pmy_mesh->UserEMFSourceTerm_;
    if(UserEMFSourceTerm != NULL) emf_sourceterms_defined = true;
}


//----------------------------------------------------------------------------------------
//! \fn void Field::CalculateCellCenteredField
//! \brief cell center B-fields are defined as spatial interpolation at the volume center

void Field::CalculateCellCenteredField(
    const FaceField &bf, AthenaArray<Real> &bc, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  // Defer to Reconstruction class to check if uniform Cartesian formula can be used
  // (unweighted average)
  const bool uniform_ave_x1 = pmy_block->precon->uniform[X1DIR];
  const bool uniform_ave_x2 = pmy_block->precon->uniform[X2DIR];
  const bool uniform_ave_x3 = pmy_block->precon->uniform[X3DIR];

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
      // calc cell centered fields first
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        const Real& b1_i   = bf.x1f(k,j,i  );
        const Real& b1_ip1 = bf.x1f(k,j,i+1);
        const Real& b2_j   = bf.x2f(k,j  ,i);
        const Real& b2_jp1 = bf.x2f(k,j+1,i);
        const Real& b3_k   = bf.x3f(k  ,j,i);
        const Real& b3_kp1 = bf.x3f(k+1,j,i);

        Real& bcc1 = bc(IB1,k,j,i);
        Real& bcc2 = bc(IB2,k,j,i);
        Real& bcc3 = bc(IB3,k,j,i);
        Real lw, rw; // linear interpolation coefficients from lower and upper cell faces

        // cell center B-fields are defined as spatial interpolation at the volume center
        if (uniform_ave_x1) {
          lw = 0.5;
          rw = 0.5;
        } else {
          const Real& x1f_i  = pco->x1f(i);
          const Real& x1f_ip = pco->x1f(i+1);
          const Real& x1v_i  = pco->x1v(i);
          const Real& dx1_i  = pco->dx1f(i);
          lw = (x1f_ip - x1v_i)/dx1_i;
          rw = (x1v_i  - x1f_i)/dx1_i;
        }
        bcc1 = lw*b1_i + rw*b1_ip1;

        if (uniform_ave_x2) {
          lw = 0.5;
          rw = 0.5;
        } else {
          const Real& x2f_j  = pco->x2f(j);
          const Real& x2f_jp = pco->x2f(j+1);
          const Real& x2v_j  = pco->x2v(j);
          const Real& dx2_j  = pco->dx2f(j);
          lw = (x2f_jp - x2v_j)/dx2_j;
          rw = (x2v_j  - x2f_j)/dx2_j;
        }
        bcc2 = lw*b2_j + rw*b2_jp1;
        if (uniform_ave_x3) {
          lw = 0.5;
          rw = 0.5;
        } else {
          const Real& x3f_k  = pco->x3f(k);
          const Real& x3f_kp = pco->x3f(k+1);
          const Real& x3v_k  = pco->x3v(k);
          const Real& dx3_k  = pco->dx3f(k);
          lw = (x3f_kp - x3v_k)/dx3_k;
          rw = (x3v_k  - x3f_k)/dx3_k;
        }
        bcc3 = lw*b3_k + rw*b3_kp1;
      }
    }
  }
  return;
}


//----------------------------------------------------------------------------------------
//! \fn void Field::AddEMFSourceTerms
//  \brief Adds source terms to emf (i.e. edge centered e)

void Field::AddEMFSourceTerms(const Real time, const Real dt,
     const AthenaArray<Real> *flux, const AthenaArray<Real> &prim,
     const AthenaArray<Real> &bcc, const AthenaArray<Real> &cons, EdgeField &e)
{
  MeshBlock *pmb = pmy_block;


  //  user-defined source terms
  if (UserEMFSourceTerm != NULL)
    UserEMFSourceTerm(pmb, time,dt,prim,bcc,cons,e);

  return;
}


void Field::RecomputeMagneticFieldFromCorrectedVectorPotential(){
    MeshBlock *pmb = pmy_block;
    int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;

    AthenaArray<Real> area;
    area.NewAthenaArray(ie+NGHOST+2);

    AthenaArray<Real> &a_x_edges = e.x1e, &a_y_edges = e.x2e, &a_z_edges = e.x3e;



      // Set B^1
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je; ++j) {
          pmb->pcoord->Face1Area(k,   j,   is, ie+1, area);
          for (int i = is; i <= ie+1; ++i) {

            //d Az /dy

            Real lenm = pmb->pcoord->GetEdge3Length(k,j,i);
            Real lenp = pmb->pcoord->GetEdge3Length(k,j+1,i);

            b.x1f(k,j,i) = 1.0/area(i) * (a_z_edges(k,j+1,i)*lenp - a_z_edges(k,j,i)*lenm)  ;

            //d Ay/dz


            lenm = pmb->pcoord->GetEdge2Length(k,j,i);
            lenp = pmb->pcoord->GetEdge2Length(k+1,j,i);

            b.x1f(k,j,i) -= 1.0/area(i) * (a_y_edges(k+1,j,i)*lenp - a_y_edges(k,j,i)*lenm)  ;

          }
        }
      }

      // Set B^2
      for (int k = ks; k <= ke; ++k) {
        for (int j = js; j <= je+1; ++j) {
          pmb->pcoord->Face2Area(k,   j,   is, ie, area);
          for (int i = is; i <= ie; ++i) {


            //d Ax /dz
            Real lenm = pmb->pcoord->GetEdge1Length(k,j,i);
            Real lenp = pmb->pcoord->GetEdge1Length(k+1,j,i);

            b.x2f(k,j,i) = 1.0/area(i) * (a_x_edges(k+1,j,i)*lenp - a_x_edges(k,j,i)*lenm);

            //d Az/dx
            Real Az_2,Az_1;

            lenm = pmb->pcoord->GetEdge1Length(k,j,i);
            lenp = pmb->pcoord->GetEdge1Length(k,j,i+1);

            b.x2f(k,j,i) -= 1.0/area(i) * (a_z_edges(k,j,i+1)*lenp - a_z_edges(k,j,i)*lenm) ;

                  
          }
        }
      }

      // Set B^3
      for (int k = ks; k <= ke+1; ++k) {
        for (int j = js; j <= je; ++j) {
          pmb->pcoord->Face3Area(k,   j,   is, ie, area);
          for (int i = is; i <= ie; ++i) {


            //d Ay /dx

            Real lenm = pmb->pcoord->GetEdge2Length(k,j,i);
            Real lenp = pmb->pcoord->GetEdge2Length(k,j,i+1);
                  

            b.x3f(k,j,i) = 1.0/area(i) * (a_y_edges(k,j,i+1)*lenp - a_y_edges(k,j,i)*lenm);

            //d Ax/dy

            lenm = pmb->pcoord->GetEdge1Length(k,j,i);
            lenp = pmb->pcoord->GetEdge1Length(k,j+1,i);

            b.x3f(k,j,i) -= 1.0/area(i) * (a_x_edges(k,j+1,i)*lenp - a_x_edges(k,j,i)*lenm);

            if (std::isnan(b.x3f(k,j,i))){
              fprintf(stderr,"NAN in field \n  Ax_2: %g Ax_1: %g \n", a_x_edges(k,j+1,i),a_x_edges(k,j,i));
            
            }
          }
        }
      }

    

  
      area.DeleteAthenaArray();


}

#if DEBUG_CHECKS
bool Field::CheckFieldDivergence(FaceField &b, std::string code_location){

  MeshBlock *pmb = pmy_block;

  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;

  bool bad_divergence = false;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  Coordinates *pcc = pmb->pmr->GetCoarseCoordinates();

  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        Real divb=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));

          int ci = (i-is)/2 + pmb->cis;
          int cj = (j-js)/2 + pmb->cjs;
          int ck = (k-ks)/2 + pmb->cks;

        // if (std::fabs(pcc->x1v(ci)+190.0)<0.2 && std::fabs(pcc->x2v(cj)+190.0)<0.2 &&  std::fabs(pcc->x3f(ck)-24.0)<0.2 && i==4 && j==4 && k==4){
        //   fprintf(stderr, "B above interface at %s ijk: %d %d %d B: %g \n xyz coarse : %g %g %g \n xyz fine: %g %g %g \n gid: %d\n", code_location.c_str(),i,j,k,b.x3f(k,j,i),
        //     pcc->x1v(ci),pcc->x2v(cj),pcc->x3f(ck),pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k+1), pmb->gid);
        // }   
        // if (std::fabs(pcc->x1v(i)+190.0)<0.2 && std::fabs(pcc->x2v(cj)+190.0)<0.2 &&  std::fabs(pcc->x3f(ck+1)-24.0)<0.2 && i==4 && j==4 && k==11){
        //   fprintf(stderr, "B below interface at %s ijk: %d %d %d B %g\n xyz coarse : %g %g %g \n xyz fine: %g %g %g \n gid: %d \n", code_location.c_str(),i,j,k+1,b.x3f(k+1,j,i),
        //     pcc->x1v(ci),pcc->x2v(cj),pcc->x3f(ck+1), pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3f(k+1), pmb->gid);
        // }  //zm 20 zp: 24)

        Real machine_precision = 1e-11; //static_cast<Real>(std::numeric_limits<Real>::epsilon());
        if (std::fabs(divb)>machine_precision || !(std::fabs(divb)<machine_precision) || isnan_volatile(divb) || check_nan(divb)) {
          bad_divergence=true;
          fprintf(stderr, "nonzero divergence!! at location:  %s\n xyz: %g %g %g x3 faces: %g %g \n divb: %g machine precision: %g  \n ijk: %d %d %d\n face1: %g %g face2: %g %g face3: %g %g \n bx1: %g %g bx2: %g %g bx3: %g %g \n coarse ijk: %d %d %d \n x: %g y: %g zm %g zp: %g \n",
            code_location.c_str(),
            pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), pmb->pcoord->x3f(k),pmb->pcoord->x3f(k+1),
            divb,machine_precision,i,j,k,
            face1(i+1),face1(i),face2p(i),face2m(i),face3p(i),face3m(i),
            b.x1f(k,j,i+1), b.x1f(k,j,i), b.x2f(k,j+1,i),b.x2f(k,j,i),
           b.x3f(k+1,j,i), b.x3f(k,j,i), 
           ci,cj,ck, pcc->x1v(ci),pcc->x2v(cj),pcc->x3f(ck),
           pcc->x3f(ck+1));
        }
      }
    }
  }

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();

  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (pmb->block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (pmb->block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }


  for(int k=kl; k<=ku; k++) {
    for(int j=jl; j<=ju; j++) {
      for(int i=il; i<=iu; i++) {

        Real b_var = b.x1f(k,j,i) + b.x1f(k,j,i+1) + b.x2f(k,j,i) + b.x2f(k,j+1,i) + b.x3f(k,j,i) + b.x3f(k+1,j,i);

        if (isnan_volatile(b_var) || check_nan(b_var)){
          fprintf(stderr, "NAN b in %s \n ijk: %d %d %d \n bx1: %g %g bx2: %g %g bx3: %g %g\n",code_location.c_str(), i,j,k,
            b.x1f(k,j,i),b.x1f(k,j,i+1),b.x2f(k,j,i),b.x2f(k,j+1,i),b.x3f(k,j,i),b.x3f(k+1,j,i) );

          for (int i1=0; i1<=2; ++i1) for (int i2=0; i2<=2; ++i2) for (int i3=0; i3<=2; ++i3)fprintf(stderr,"mesh refinement levels. \n Current: %g neighbor: %d i1 i2 i3: %d %d %d \n ",
          pmb->loc.level,pmb->pbval->nblevel[i1][i2][i3],i1,i2,i3);
          exit(0);
        }

        for (int n_hydro = 0; n_hydro<NHYDRO; n_hydro++) {
          if (isnan_volatile(pmb->phydro->w(n_hydro,k,j,i)) || check_nan(pmb->phydro->w(n_hydro,k,j,i))){
            fprintf(stderr,"NAN hydro variable!! in %s \n ijk: %d %d %d \n n_hydro: %d  den: %g press: %g v: %g %g %g \n bx: %g %g by: %g %g bz: %g %g \n bcc: %g %g %g \n",
              code_location.c_str(),i,j,k,n_hydro,
              pmb->phydro->w(IDN,k,j,i), pmb->phydro->w(IPR,k,j,i),pmb->phydro->w(IVX,k,j,i),
              pmb->phydro->w(IVY,k,j,i),pmb->phydro->w(IVZ,k,j,i), 
              b.x1f(k,j,i),b.x1f(k,j,i+1),b.x2f(k,j,i),b.x2f(k,j+1,i),b.x3f(k,j,i),b.x3f(k+1,j,i),
              bcc(IB1,k,j,i), bcc(IB2,k,j,i), bcc(IB3,k,j,i));
            exit(0);
          }
        }
      }
    }
  }

  return bad_divergence;
}

#endif