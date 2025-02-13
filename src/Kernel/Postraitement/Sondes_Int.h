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

#ifndef Sondes_Int_included
#define Sondes_Int_included

#include <TRUST_List.h>
#include <Sonde_Int.h>

/*! @brief classe Sondes_Int Cette classe represente une liste d'objets de type Sonde_Int.
 *
 * @sa Sonde_Int Postraitement
 */

class Sondes_Int : public LIST(Sonde_Int)
{
  Declare_instanciable(Sondes_Int);
public:
  inline void ouvrir_fichiers();
  inline void fermer_fichiers();
  void associer_post(const Postraitement&);
  void postraiter(double );
  void mettre_a_jour(double temps, double tinit);

private:
  OBS_PTR(Postraitement) mon_post;
};

/*! @brief Ouvre tous les fichiers associes a chacune des sondes de la liste.
 *
 */
inline void Sondes_Int::ouvrir_fichiers()
{
  for (auto &itr : *this) itr.ouvrir_fichier();
}

/*! @brief Ferme tous les fichiers des sondes de la liste.
 *
 */
inline void Sondes_Int::fermer_fichiers()
{
  for (auto &itr : *this) itr.fermer_fichier();
}

#endif /* Sondes_Int_included */
