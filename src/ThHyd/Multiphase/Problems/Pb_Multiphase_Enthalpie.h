/****************************************************************************
* Copyright (c) 2024, CEA
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
* IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
* OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*****************************************************************************/

#ifndef Pb_Multiphase_Enthalpie_included
#define Pb_Multiphase_Enthalpie_included

#include <Energie_Multiphase_Enthalpie.h>
#include <Pb_Multiphase.h>

class Pb_Multiphase_Enthalpie : public Pb_Multiphase
{
  Declare_instanciable(Pb_Multiphase_Enthalpie);
public:
  inline bool resolution_en_T() const override { return false; /* en h */ }
  void associer_milieu_base(const Milieu_base& ) override;
  int verifier() override;

  const Equation_base& equation(int i) const override { return (i == 2) ? eq_energie_enthalpie_ : Pb_Multiphase::equation(i); }
  Equation_base& equation(int i) override { return (i == 2) ? eq_energie_enthalpie_ : Pb_Multiphase::equation(i); }

  Equation_base& equation_energie() override { return eq_energie_enthalpie_; }
  const Equation_base& equation_energie() const override { return eq_energie_enthalpie_; }

protected:
  Energie_Multiphase_Enthalpie eq_energie_enthalpie_;
};

#endif /* Pb_Multiphase_Enthalpie_included*/
