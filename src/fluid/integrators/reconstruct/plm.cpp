//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================

// Primary header
#include "../fluid_integrator.hpp"

// Athena headers
#include "../../../athena.hpp"         // macros, Real
#include "../../../athena_arrays.hpp"  // AthenaArray
#include "../../fluid.hpp"          // Fluid
#include "../../../mesh.hpp"           // Block

//======================================================================================
//! \file plm.cpp
//  \brief  piecewise linear reconstruction
//======================================================================================

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX1()
//  \brief 

void FluidIntegrator::PiecewiseLinearX1(const int k, const int j,
  const int il, const int iu,
  const AthenaArray<Real> &q, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  Real dql,dqr,dqc,q_im1,q_i;
  for (int n=0; n<NWAVE; ++n) {
    if (n==NFLUID){
      for (int i=il; i<=iu; ++i){
        Real& dx_im2 = pmy_fluid->pmy_block->dx1v(i-2);
        Real& dx_im1 = pmy_fluid->pmy_block->dx1v(i-1);
        Real& dx_i   = pmy_fluid->pmy_block->dx1v(i);

        q_im1 = bcc(IB2,k,j,i-1);
        q_i   = bcc(IB2,k,j,i  );
        dql = (bcc(IB2,k,j,i-1) - bcc(IB2,k,j,i-2))/dx_im2;
        dqc = (bcc(IB2,k,j,i  ) - bcc(IB2,k,j,i-1))/dx_im1;
        dqr = (bcc(IB2,k,j,i+1) - bcc(IB2,k,j,i  ))/dx_i;
        // compute ql_(i-1/2) using Mignone 2014's modified van-Leer limiter
        Real dq2 = dql*dqc;
        ql(n,i) = q_im1;
        if(dq2>0.0) {
          Real dxfr=pmy_fluid->pmy_block->x1f(i)-pmy_fluid->pmy_block->x1v(i-1);
          Real cf=dx_im1/dxfr;
          Real cb=dx_im2/(pmy_fluid->pmy_block->x1v(i-1)-pmy_fluid->pmy_block->x1f(i-1));
          ql(n,i) += dxfr*dq2*(cf*dql+cb*dqc)/(dql*dql+(cf+cb-2.0)*dq2+dqc*dqc);
        }

        // compute qr_(i-1/2) using Mignone 2014's modified van-Leer limiter
        dq2 = dqc*dqr;
        qr(n,i) = q_i;
        if(dq2>0.0) {
          Real dxfl=pmy_fluid->pmy_block->x1v(i)-pmy_fluid->pmy_block->x1f(i);
          Real cf=dx_i/(pmy_fluid->pmy_block->x1f(i+1)-pmy_fluid->pmy_block->x1v(i));
          Real cb=dx_im1/dxfl;
          qr(n,i) -= dxfl*dq2*(cf*dqc+cb*dqr)/(dqc*dqc+(cf+cb-2.0)*dq2+dqr*dqr);
        }
      }
    } else if (n==(NFLUID+1)) {
      for (int i=il; i<=iu; ++i){
        Real& dx_im2 = pmy_fluid->pmy_block->dx1v(i-2);
        Real& dx_im1 = pmy_fluid->pmy_block->dx1v(i-1);
        Real& dx_i   = pmy_fluid->pmy_block->dx1v(i);

        q_im1 = bcc(IB3,k,j,i-1);
        q_i   = bcc(IB3,k,j,i  );
        dql = (bcc(IB3,k,j,i-1) - bcc(IB3,k,j,i-2))/dx_im2;
        dqc = (bcc(IB3,k,j,i  ) - bcc(IB3,k,j,i-1))/dx_im1;
        dqr = (bcc(IB3,k,j,i+1) - bcc(IB3,k,j,i  ))/dx_i;
        // compute ql_(i-1/2) using Mignone 2014's modified van-Leer limiter
        Real dq2 = dql*dqc;
        ql(n,i) = q_im1;
        if(dq2>0.0) {
          Real dxfr=pmy_fluid->pmy_block->x1f(i)-pmy_fluid->pmy_block->x1v(i-1);
          Real cf=dx_im1/dxfr;
          Real cb=dx_im2/(pmy_fluid->pmy_block->x1v(i-1)-pmy_fluid->pmy_block->x1f(i-1));
          ql(n,i) += dxfr*dq2*(cf*dql+cb*dqc)/(dql*dql+(cf+cb-2.0)*dq2+dqc*dqc);
        }

        // compute qr_(i-1/2) using Mignone 2014's modified van-Leer limiter
        dq2 = dqc*dqr;
        qr(n,i) = q_i;
        if(dq2>0.0) {
          Real dxfl=pmy_fluid->pmy_block->x1v(i)-pmy_fluid->pmy_block->x1f(i);
          Real cf=dx_i/(pmy_fluid->pmy_block->x1f(i+1)-pmy_fluid->pmy_block->x1v(i));
          Real cb=dx_im1/dxfl;
          qr(n,i) -= dxfl*dq2*(cf*dqc+cb*dqr)/(dqc*dqc+(cf+cb-2.0)*dq2+dqr*dqr);
        }
      }
    } else {
      for (int i=il; i<=iu; ++i){
        Real& dx_im2 = pmy_fluid->pmy_block->dx1v(i-2);
        Real& dx_im1 = pmy_fluid->pmy_block->dx1v(i-1);
        Real& dx_i   = pmy_fluid->pmy_block->dx1v(i);

        q_im1 = q(n,k,j,i-1);
        q_i   = q(n,k,j,i  );
        dql = (q(n,k,j,i-1) - q(n,k,j,i-2))/dx_im2;
        dqc = (q(n,k,j,i  ) - q(n,k,j,i-1))/dx_im1;
        dqr = (q(n,k,j,i+1) - q(n,k,j,i  ))/dx_i;
        // compute ql_(i-1/2) using Mignone 2014's modified van-Leer limiter
        Real dq2 = dql*dqc;
        ql(n,i) = q_im1;
        if(dq2>0.0) {
          Real dxfr=pmy_fluid->pmy_block->x1f(i)-pmy_fluid->pmy_block->x1v(i-1);
          Real cf=dx_im1/dxfr;
          Real cb=dx_im2/(pmy_fluid->pmy_block->x1v(i-1)-pmy_fluid->pmy_block->x1f(i-1));
          ql(n,i) += dxfr*dq2*(cf*dql+cb*dqc)/(dql*dql+(cf+cb-2.0)*dq2+dqc*dqc);
        }

        // compute qr_(i-1/2) using Mignone 2014's modified van-Leer limiter
        dq2 = dqc*dqr;
        qr(n,i) = q_i;
        if(dq2>0.0) {
          Real dxfl=pmy_fluid->pmy_block->x1v(i)-pmy_fluid->pmy_block->x1f(i);
          Real cf=dx_i/(pmy_fluid->pmy_block->x1f(i+1)-pmy_fluid->pmy_block->x1v(i));
          Real cb=dx_im1/dxfl;
          qr(n,i) -= dxfl*dq2*(cf*dqc+cb*dqr)/(dqc*dqc+(cf+cb-2.0)*dq2+dqr*dqr);
        }
      }
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX2()
//  \brief 

void FluidIntegrator::PiecewiseLinearX2(const int k, const int j,
  const int il, const int iu,
  const AthenaArray<Real> &q, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  Real dx2jm2i = 1.0/pmy_fluid->pmy_block->dx2v(j-2);
  Real dx2jm1i = 1.0/pmy_fluid->pmy_block->dx2v(j-1);
  Real dx2ji   = 1.0/pmy_fluid->pmy_block->dx2v(j);
  Real dxfr=pmy_fluid->pmy_block->x2f(j)-pmy_fluid->pmy_block->x2v(j-1);
  Real dxfl=pmy_fluid->pmy_block->x2v(j)-pmy_fluid->pmy_block->x2f(j);
  Real cfm=pmy_fluid->pmy_block->dx2v(j-1)/dxfr;
  Real cbm=pmy_fluid->pmy_block->dx2v(j-2)/(pmy_fluid->pmy_block->x2v(j-1)-pmy_fluid->pmy_block->x2f(j-1));
  Real cfp=pmy_fluid->pmy_block->dx2v(j)/(pmy_fluid->pmy_block->x2f(j+1)-pmy_fluid->pmy_block->x2v(j));
  Real cbp=pmy_fluid->pmy_block->dx2v(j-1)/dxfl;
  Real dql,dqr,dqc,q_jm1,q_j;

  for (int n=0; n<NWAVE; ++n) {
    if (n==NFLUID){
      for (int i=il; i<=iu; ++i){
        q_jm1 = bcc(IB3,k,j-1,i);
        q_j   = bcc(IB3,k,j  ,i);
        dql = (bcc(IB3,k,j-1,i) - bcc(IB3,k,j-2,i))*dx2jm2i;
        dqc = (bcc(IB3,k,j  ,i) - bcc(IB3,k,j-1,i))*dx2jm1i;
        dqr = (bcc(IB3,k,j+1,i) - bcc(IB3,k,j  ,i))*dx2ji;
        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_jm1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
        
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_j;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    } else if (n==(NFLUID+1)) {
      for (int i=il; i<=iu; ++i){
        q_jm1 = bcc(IB1,k,j-1,i);
        q_j   = bcc(IB1,k,j  ,i);
        dql = (bcc(IB1,k,j-1,i) - bcc(IB1,k,j-2,i))*dx2jm2i;
        dqc = (bcc(IB1,k,j  ,i) - bcc(IB1,k,j-1,i))*dx2jm1i;
        dqr = (bcc(IB1,k,j+1,i) - bcc(IB1,k,j  ,i))*dx2ji;
        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_jm1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
        
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_j;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    } else {
      for (int i=il; i<=iu; ++i){
        q_jm1 = q(n,k,j-1,i);
        q_j   = q(n,k,j  ,i);
        dql = (q(n,k,j-1,i) - q(n,k,j-2,i))*dx2jm2i;
        dqc = (q(n,k,j  ,i) - q(n,k,j-1,i))*dx2jm1i;
        dqr = (q(n,k,j+1,i) - q(n,k,j  ,i))*dx2ji;

        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_jm1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
        
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_j;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    }
  }

  return;
}

//--------------------------------------------------------------------------------------
//! \fn FluidIntegrator::ReconstructionFuncX3()
//  \brief 

void FluidIntegrator::PiecewiseLinearX3(const int k, const int j,
  const int il, const int iu,
  const AthenaArray<Real> &q, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &ql, AthenaArray<Real> &qr)
{
  Real dx3km2i = 1.0/pmy_fluid->pmy_block->dx3v(k-2);
  Real dx3km1i = 1.0/pmy_fluid->pmy_block->dx3v(k-1);
  Real dx3ki   = 1.0/pmy_fluid->pmy_block->dx3v(k);
  Real dxfr=pmy_fluid->pmy_block->x3f(k)-pmy_fluid->pmy_block->x3v(k-1);
  Real dxfl=pmy_fluid->pmy_block->x3v(k)-pmy_fluid->pmy_block->x3f(k);
  Real cfm=pmy_fluid->pmy_block->dx3v(k-1)/dxfr;
  Real cbm=pmy_fluid->pmy_block->dx3v(k-2)/(pmy_fluid->pmy_block->x3v(k-1)-pmy_fluid->pmy_block->x3f(k-1));
  Real cfp=pmy_fluid->pmy_block->dx3v(k)/(pmy_fluid->pmy_block->x3f(k+1)-pmy_fluid->pmy_block->x3v(k));
  Real cbp=pmy_fluid->pmy_block->dx3v(k-1)/dxfl;
  Real dql,dqr,dqc,q_km1,q_k;

  for (int n=0; n<NWAVE; ++n) {
    if (n==NFLUID){
      for (int i=il; i<=iu; ++i){
        q_km1 = bcc(IB1,k-1,j,i);
        q_k   = bcc(IB1,k  ,j,i);
        dql = (bcc(IB1,k-1,j,i) - bcc(IB1,k-2,j,i))*dx3km2i;
        dqc = (bcc(IB1,k  ,j,i) - bcc(IB1,k-1,j,i))*dx3km1i;
        dqr = (bcc(IB1,k+1,j,i) - bcc(IB1,k  ,j,i))*dx3ki;

        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_km1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
      
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_k;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    } else if (n==(NFLUID+1)) {
      for (int i=il; i<=iu; ++i){
        q_km1 = bcc(IB2,k-1,j,i);
        q_k   = bcc(IB2,k  ,j,i);
        dql = (bcc(IB2,k-1,j,i) - bcc(IB2,k-2,j,i))*dx3km2i;
        dqc = (bcc(IB2,k  ,j,i) - bcc(IB2,k-1,j,i))*dx3km1i;
        dqr = (bcc(IB2,k+1,j,i) - bcc(IB2,k  ,j,i))*dx3ki;

        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_km1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
      
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_k;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    } else {
      for (int i=il; i<=iu; ++i){
        q_km1 = q(n,k-1,j,i);
        q_k   = q(n,k  ,j,i);
        dql = (q(n,k-1,j,i) - q(n,k-2,j,i))*dx3km2i;
        dqc = (q(n,k  ,j,i) - q(n,k-1,j,i))*dx3km1i;
        dqr = (q(n,k+1,j,i) - q(n,k  ,j,i))*dx3ki;

        // Apply monotonicity constraints, compute ql_(i-1/2)
        Real dq2 = dql*dqc;
        ql(n,i) = q_km1;
        if(dq2>0.0)
          ql(n,i) += dxfr*dq2*(cfm*dql+cbm*dqc)/(dql*dql+(cfm+cbm-2.0)*dq2+dqc*dqc);
      
        // Apply monotonicity constraints, compute qr_(i-1/2)
        dq2 = dqc*dqr;
        qr(n,i) = q_k;
        if(dq2>0.0)
          qr(n,i) -= dxfl*dq2*(cfp*dqc+cbp*dqr)/(dqc*dqc+(cfp+cbp-2.0)*dq2+dqr*dqr);
      }
    }
  }

  return;
}
