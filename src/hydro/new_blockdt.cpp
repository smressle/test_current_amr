//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file new_blockdt.cpp
//! \brief computes timestep using CFL condition on a MEshBlock

// C headers

// C++ headers
#include <algorithm>  // min()
#include <cmath>      // fabs(), sqrt()
#include <limits>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../field/field_diffusion/field_diffusion.hpp"
#include "../mesh/mesh.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../scalars/scalars.hpp"
#include "hydro.hpp"
#include "hydro_diffusion/hydro_diffusion.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


Real max_wave_speed_gr(int DIR, int i, int j, int k,MeshBlock *pmb,AthenaArray<Real> &w,AthenaArray<Real> &g_,AthenaArray<Real> &gi_,AthenaArray<Real> &bcc,FaceField b ) {
  Real Acov[4],Acon[4],Bcon[4],Bcov[4];

  for (int mu=0; mu<=4; ++mu){
    Acov[mu] = 0.0;
    Bcov[mu] = 0.0;
    Acon[mu] = 0.0;
    Bcon[mu] = 0.0;
  }

  Acov[DIR] = 1.0;
  Bcov[0] = 1.0;


  Acon[0] = gi_(I00,i)*Acov[0] + gi_(I01,i)*Acov[1] + gi_(I02)*Acov[2] + gi_(I03)*Acov[3];
  Acon[1] = gi_(I01,i)*Acov[0] + gi_(I11,i)*Acov[1] + gi_(I12)*Acov[2] + gi_(I13)*Acov[3];
  Acon[2] = gi_(I02,i)*Acov[0] + gi_(I12,i)*Acov[1] + gi_(I22)*Acov[2] + gi_(I23)*Acov[3];
  Acon[3] = gi_(I03,i)*Acov[0] + gi_(I13,i)*Acov[1] + gi_(I23)*Acov[2] + gi_(I33)*Acov[3];

  Bcon[0] = gi_(I00,i)*Bcov[0] + gi_(I01,i)*Bcov[1] + gi_(I02,i)*Bcov[2] + gi_(I03,i)*Bcov[3];
  Bcon[1] = gi_(I01,i)*Bcov[0] + gi_(I11,i)*Bcov[1] + gi_(I12,i)*Bcov[2] + gi_(I13,i)*Bcov[3];
  Bcon[2] = gi_(I02,i)*Bcov[0] + gi_(I12,i)*Bcov[1] + gi_(I22,i)*Bcov[2] + gi_(I23,i)*Bcov[3];
  Bcon[3] = gi_(I03,i)*Bcov[0] + gi_(I13,i)*Bcov[1] + gi_(I23,i)*Bcov[2] + gi_(I33,i)*Bcov[3];

  Real Asq = Acon[0]*Acov[0] + Acon[1]*Acov[1] + Acon[2]*Acov[2] + Acon[3]*Acov[3];
  Real Bsq = Bcon[0]*Bcov[0] + Bcon[1]*Bcov[1] + Bcon[2]*Bcov[2] + Bcon[3]*Bcov[3];


  Real uu1 = w(IVX,k,j,i);
  Real uu2 = w(IVY,k,j,i);
  Real uu3 = w(IVZ,k,j,i);
  Real tmp = g_(I11,i)*uu1*uu1 + 2.0*g_(I12,i)*uu1*uu2 + 2.0*g_(I13,i)*uu1*uu3
           + g_(I22,i)*uu2*uu2 + 2.0*g_(I23,i)*uu2*uu3
           + g_(I33,i)*uu3*uu3;
  Real gamma = std::sqrt(1.0 + tmp);

  // Calculate 4-velocity
  Real alpha = std::sqrt(-1.0/gi_(I00,i));
  Real u0 = gamma/alpha;
  Real u1 = uu1 - alpha * gamma * gi_(I01,i);
  Real u2 = uu2 - alpha * gamma * gi_(I02,i);
  Real u3 = uu3 - alpha * gamma * gi_(I03,i);

  Real Au = u0*Acov[0] + u1*Acov[1] + u2*Acov[2] + u3*Acov[3];


  Real Bu = u0*Bcov[0] + u1*Bcov[1] + u2*Bcov[2] + u3*Bcov[3];

  Real AB = Acon[0]*Bcov[0] + Acon[1]*Bcov[1] + Acon[2]*Bcov[2] + Acon[3]*Bcov[3];

  Real Au2 = Au*Au;
  Real Bu2 = Bu*Bu;
  Real AuBu = Au*Bu;

  Real b_sq;

  if (MAGNETIC_FIELDS_ENABLED) {


    Real u_0,u_1,u_2,u_3;
    pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

    // Calculate 4-magnetic field
    Real bb1 = bcc(IB1,k,j,i); /// + std::abs(b_x1f(k,j,i) - bcc(IB1,k,j,i));
    Real bb2 = bcc(IB2,k,j,i);
    Real bb3 = bcc(IB3,k,j,i);

    if (DIR==1) bb1 += std::abs(b.x1f(k,j,i) - bcc(IB1,k,j,i));
    if (DIR==2) bb2 += std::abs(b.x2f(k,j,i) - bcc(IB2,k,j,i));
    if (DIR==3) bb3 += std::abs(b.x3f(k,j,i) - bcc(IB3,k,j,i));

    Real b0 = g_(I01,i)*u0*bb1 + g_(I02,i)*u0*bb2 + g_(I03,i)*u0*bb3
            + g_(I11,i)*u1*bb1 + g_(I12,i)*u1*bb2 + g_(I13,i)*u1*bb3
            + g_(I12,i)*u2*bb1 + g_(I22,i)*u2*bb2 + g_(I23,i)*u2*bb3
            + g_(I13,i)*u3*bb1 + g_(I23,i)*u3*bb2 + g_(I33,i)*u3*bb3;
    Real b1 = (bb1 + b0 * u1) / u0;
    Real b2 = (bb2 + b0 * u2) / u0;
    Real b3 = (bb3 + b0 * u3) / u0;
    Real b_0, b_1, b_2, b_3;
    pmb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

    // Calculate bsq
    b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

  }
  else{
    b_sq = 0.0;
  }
  Real gam = pmb->peos->GetGamma();
  // Find fast magnetosonic speed
  Real rho = w(IDN,k,j,i);
  Real u = w(IPR,k,j,i) / (gam-1.0); 
  Real ef = rho + gam*u;
  Real ee = b_sq + ef;
  Real va2 = b_sq/ee;
  Real cs2 = gam*(gam - 1.)*u/ef;

  Real cms2 = cs2 + va2 - cs2*va2;

  Real SMALL = 1e-10;

  cms2 = (cms2 < 0) ? SMALL : cms2;
  cms2 = (cms2 > 1) ? 1 : cms2;

  // Require that speed of wave measured by observer q->ucon is cms2

  Real A = Bu2 - (Bsq + Bu2)*cms2;
  Real B = 2.*(AuBu - (AB + AuBu)*cms2);
  Real C = Au2 - (Asq + Au2)*cms2;

  Real discr = B*B - 4.*A*C;
  discr = (discr < 0.) ? 0. : discr;
  discr = std::sqrt(discr);

  Real vp = -(-B + discr)/(2.*A);
  Real vm = -(-B - discr)/(2.*A);

  Real cmax = (vp > vm) ? vp : vm;
  Real cmin = (vp > vm) ? vm : vp;

    // (*cmax)[k][j][i] = fabs(MY_MAX(MY_MAX(0., (*cmaxL)[k][j][i]), (*cmaxR)[k][j][i]));
    // (*cmin)[k][j][i] = fabs(MY_MAX(MY_MAX(0., -(*cminL)[k][j][i]), -(*cminR)[k][j][i]));
    // (*ctop)[dir][k][j][i] = MY_MAX((*cmax)[k][j][i], (*cmin)[k][j][i]);
  Real ctop = std::max(std::fabs(cmax),std::fabs(cmin));

  if (ctop>2)
  fprintf(stderr,"dir: %d ijk: %d %d %d \n xyz: %g %g %g \n cms2: %g ABC: %g %g %g \n Bu2: %g Au2: %g vp: %g vm: %g \n ctop: %g \n",DIR, i,j,k, pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k),cms2,A,B,C,Bu2,Au2,vp,vm,ctop);

  return ctop;



}

//----------------------------------------------------------------------------------------
//! \fn void Hydro::NewBlockTimeStep()
//! \brief calculate the minimum timestep within a MeshBlock

void Hydro::NewBlockTimeStep() {
  MeshBlock *pmb = pmy_block;
  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  // hyperbolic timestep constraint in each (x1-slice) cell along coordinate direction:
  AthenaArray<Real> &dt1 = dt1_, &dt2 = dt2_, &dt3 = dt3_;  // (x1 slices)
  Real wi[NWAVE];

  Real real_max = std::numeric_limits<Real>::max();
  Real min_dt = real_max;
  // Note, "dt_hyperbolic" currently refers to the dt limit imposed by evoluiton of the
  // ideal hydro or MHD fluid by the main integrator (even if not strictly hyperbolic)
  Real min_dt_hyperbolic  = real_max;
  // TODO(felker): consider renaming dt_hyperbolic after general execution model is
  // implemented and flexibility from #247 (zero fluid configurations) is
  // addressed. dt_hydro, dt_main (inaccurate since "dt" is actually main), dt_MHD?
  Real min_dt_parabolic  = real_max;
  Real min_dt_user  = real_max;

  // TODO(felker): skip this next loop if pm->fluid_setup == FluidFormulation::disabled
  FluidFormulation fluid_status = pmb->pmy_mesh->fluid_setup;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      pmb->pcoord->CenterWidth1(k, j, is, ie, dt1);
      pmb->pcoord->CenterWidth2(k, j, is, ie, dt2);
      pmb->pcoord->CenterWidth3(k, j, is, ie, dt3);
      if (!RELATIVISTIC_DYNAMICS) {
#pragma ivdep
        for (int i=is; i<=ie; ++i) {
          wi[IDN] = w(IDN,k,j,i);
          wi[IVX] = w(IVX,k,j,i);
          wi[IVY] = w(IVY,k,j,i);
          wi[IVZ] = w(IVZ,k,j,i);
          if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);
          if (fluid_status == FluidFormulation::evolve) {
            if (MAGNETIC_FIELDS_ENABLED) {
              AthenaArray<Real> &bcc = pmb->pfield->bcc, &b_x1f = pmb->pfield->b.x1f,
                              &b_x2f = pmb->pfield->b.x2f, &b_x3f = pmb->pfield->b.x3f;
              Real bx = bcc(IB1,k,j,i) + std::abs(b_x1f(k,j,i) - bcc(IB1,k,j,i));
              wi[IBY] = bcc(IB2,k,j,i);
              wi[IBZ] = bcc(IB3,k,j,i);
              Real cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              dt1(i) /= (std::abs(wi[IVX]) + cf);

              wi[IBY] = bcc(IB3,k,j,i);
              wi[IBZ] = bcc(IB1,k,j,i);
              bx = bcc(IB2,k,j,i) + std::abs(b_x2f(k,j,i) - bcc(IB2,k,j,i));
              cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              dt2(i) /= (std::abs(wi[IVY]) + cf);

              wi[IBY] = bcc(IB1,k,j,i);
              wi[IBZ] = bcc(IB2,k,j,i);
              bx = bcc(IB3,k,j,i) + std::abs(b_x3f(k,j,i) - bcc(IB3,k,j,i));
              cf = pmb->peos->FastMagnetosonicSpeed(wi,bx);
              dt3(i) /= (std::abs(wi[IVZ]) + cf);
            } else {
              Real cs = pmb->peos->SoundSpeed(wi);
              dt1(i) /= (std::abs(wi[IVX]) + cs);
              dt2(i) /= (std::abs(wi[IVY]) + cs);
              dt3(i) /= (std::abs(wi[IVZ]) + cs);
            }
          } else { // FluidFormulation::background or disabled. Assume scalar advection:
            dt1(i) /= (std::abs(wi[IVX]));
            dt2(i) /= (std::abs(wi[IVY]));
            dt3(i) /= (std::abs(wi[IVZ]));
          }
        }
      }


      if (RELATIVISTIC_DYNAMICS){
        pmb->pcoord->CellMetric(k,j,is,ie,g_,gi_); 
        #pragma ivdep
        for (int i=is; i<=ie; ++i) {


          // Real cl1 = ( -g_(I01,i) + std::sqrt( SQR(g_(I01,i)) - g_(I00,i)*g_(I11,i) ) ) / g_(I11,i);
          // Real cl2 = ( -g_(I02,i) + std::sqrt( SQR(g_(I02,i)) - g_(I00,i)*g_(I22,i) ) ) / g_(I22,i);
          // Real cl3 = ( -g_(I03,i) + std::sqrt( SQR(g_(I03,i)) - g_(I00,i)*g_(I33,i) ) ) / g_(I33,i);

          // Real cl1 = max_wave_speed_gr(1,i,j,k,pmb,w,g_,gi_,pmb->pfield->bcc,pmb->pfield->b);
          // Real cl2 = max_wave_speed_gr(2,i,j,k,pmb,w,g_,gi_,pmb->pfield->bcc,pmb->pfield->b);
          // Real cl3 = max_wave_speed_gr(3,i,j,k,pmb,w,g_,gi_,pmb->pfield->bcc,pmb->pfield->b);

          // dt1(i) /= cl1;
          // dt2(i) /= cl2;
          // dt3(i) /= cl3;
        }
      }

      // compute minimum of (v1 +/- C)
      for (int i=is; i<=ie; ++i) {
        Real& dt_1 = dt1(i);
        min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_1);
      }

      // if grid is 2D/3D, compute minimum of (v2 +/- C)
      if (pmb->block_size.nx2 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_2 = dt2(i);
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_2);
        }
      }

      // if grid is 3D, compute minimum of (v3 +/- C)
      if (pmb->block_size.nx3 > 1) {
        for (int i=is; i<=ie; ++i) {
          Real& dt_3 = dt3(i);
          min_dt_hyperbolic = std::min(min_dt_hyperbolic, dt_3);
        }
      }
    }
  }

  // calculate the timestep limited by the diffusion processes
  if (hdif.hydro_diffusion_defined) {
    Real min_dt_vis, min_dt_cnd;
    hdif.NewDiffusionDt(min_dt_vis, min_dt_cnd);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_vis);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_cnd);
  } // hydro diffusion

  if (MAGNETIC_FIELDS_ENABLED &&
      pmb->pfield->fdif.field_diffusion_defined) {
    Real min_dt_oa, min_dt_hall;
    pmb->pfield->fdif.NewDiffusionDt(min_dt_oa, min_dt_hall);
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_oa);
    // Hall effect is dispersive, not diffusive:
    min_dt_hyperbolic = std::min(min_dt_hyperbolic, min_dt_hall);
  } // field diffusion

  if (NSCALARS > 0 && pmb->pscalars->scalar_diffusion_defined) {
    Real min_dt_scalar_diff = pmb->pscalars->NewDiffusionDt();
    min_dt_parabolic = std::min(min_dt_parabolic, min_dt_scalar_diff);
  } // passive scalar diffusion

  min_dt_hyperbolic *= pmb->pmy_mesh->cfl_number;
  // scale the theoretical stability limit by a safety factor = the hyperbolic CFL limit
  // (user-selected or automaticlaly enforced). May add independent parameter "cfl_diff"
  // in the future (with default = cfl_number).
  min_dt_parabolic *= pmb->pmy_mesh->cfl_number;

  // For orbital advection, give a restriction on dt_hyperbolic.
  if (pmb->porb->orbital_advection_active) {
    Real min_dt_orb = pmb->porb->NewOrbitalAdvectionDt();
    min_dt_hyperbolic = std::min(min_dt_hyperbolic, min_dt_orb);
  }

  // set main integrator timestep as the minimum of the appropriate timestep constraints:
  // hyperbolic: (skip if fluid is nonexistent or frozen)
  min_dt = std::min(min_dt, min_dt_hyperbolic);
  // user:
  if (UserTimeStep_ != nullptr) {
    min_dt_user = UserTimeStep_(pmb);
    min_dt = std::min(min_dt, min_dt_user);
  }
  // parabolic:
  // STS handles parabolic terms -> then take the smaller of hyperbolic or user timestep
  if (!STS_ENABLED) {
    // otherwise, take the smallest of the hyperbolic, parabolic, user timesteps
    min_dt = std::min(min_dt, min_dt_parabolic);
  }
  pmb->new_block_dt_ = min_dt;
  pmb->new_block_dt_hyperbolic_ = min_dt_hyperbolic;
  pmb->new_block_dt_parabolic_ = min_dt_parabolic;
  pmb->new_block_dt_user_ = min_dt_user;

  return;
}
