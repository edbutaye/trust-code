/****************************************************************************
* Copyright (c) 2015 - 2016, CEA
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
//////////////////////////////////////////////////////////////////////////////
//
// File:        ArrOfDouble.cpp
// Directory:   $TRUST_ROOT/src/Kernel/Math
// Version:     /main/10
//
//////////////////////////////////////////////////////////////////////////////
//
// WARNING: DO NOT EDIT THIS FILE! Only edit the template file ArrOfDouble.cpp.P
//
//
#include <ArrOfDouble.h>
//#include <FArrOfDouble.h>
#include <math.h>
#ifdef SGI_
#include <bstring.h>
#endif
#include <Memoire.h>
#include <Vect_ArrOfDouble.h>
#include <Ref_ArrOfDouble.h>
#include <climits>

Implemente_instanciable_sans_constructeur_ni_destructeur(ArrOfDouble,"ArrOfDouble",Array_base);

Implemente_vect(ArrOfDouble);

Implemente_ref(ArrOfDouble);

// ******************************************************************
//
//             Implementation des methodes de VDoubledata
//
// ******************************************************************
////////////////////////////////////////////////////////////////////
//
// .DESCRIPTION
// VDoubledata alloue une zone de memoire de la taille specifiee au
// constructeur, et libere la zone de memoire a la destruction.
// La memoire peut etre allouee sur le tas (avec new) ou par le
// mecanisme Memoire::add_trav_double.
//
// "ref_count" compte le nombre de pointeurs qui font reference a "this".
// (permet au dernier utilisateur de l'objet de le detruire), voir
// ArrOfDouble.
//
// .SECTION voir aussi
///////////////////////////////////////////////////////////////////

class VDoubledata
{
public:
  VDoubledata(int size, ArrOfDouble::Storage storage);
  ~VDoubledata();
  int          add_one_ref();
  int          suppr_one_ref();
  double *        get_data();
  const double *  get_data() const;
  inline int   ref_count() const;
  inline int   get_size() const;
private:
  // Le constructeur par copie et l'operateur= sont interdits.
  VDoubledata(const VDoubledata& v);
  VDoubledata& operator=(const VDoubledata& v);

  // "data" est un pointeur sur une zone de memoire de taille
  // sz * sizeof(double), allouee par le
  // constructeur et liberee par le destructeur.
  // Ce pointeur n'est jamais nul meme si size_==0
  double * data_;
  // Compteur incremente par add_one_ref et decremente par suppr_one_ref.
  // Contient le nombre d'objets ArrOfDouble dont le membre "p" pointe
  // vers "this". On a ref_count_ >= 0.
  int ref_count_;
  // "sz" est la taille du tableau "data_" alloue
  // On a sz >= 0.
  int size_;
  // Si storage est de type TEMP_STORAGE, d_ptr_trav porte la reference
  // a la zone allouee, sinon le pointeur est nul.
  Double_ptr_trav * d_ptr_trav_;
};


// Description:
//    Construit un VDoubledata de taille size >= 0
// Parametre: int s
//    Signification: taille du VDoubledata, il faut size >= 0
// Parametre: Storage storage
//    Signification: indique si la memoire doit etre allouee
//                   avec "new" ou avec "memoire.add_trav_double()"
//    Valeurs par defaut: STANDARD (allocation avec "new")
// Postcondition:
//    data_ n'est jamais nul, meme si size==0
VDoubledata::VDoubledata(int size, ArrOfDouble::Storage storage)
{
  int size_warning=100000000;
  assert(size >= 0);
  if (size == 0)
    storage = ArrOfDouble::STANDARD;

  switch (storage)
    {
    case ArrOfDouble::STANDARD:
      {
        d_ptr_trav_ = 0;
        if (size>size_warning) Cerr << "Warning: Allocating an array of " << size << " Double ...";
#ifdef _EXCEPTION_
        // Allocation de la memoire sur le tas
        try
          {
            data_ = new double[size];
          }
        catch(...)
          {
            Cerr << "unable to allocate " << size << " double " << finl;
            Process::exit();
          }
#else
        data_ = new double[size];
        if(!data_)
          {
            Cerr << "unable to allocate " << size << "double " << finl;
            Process::exit();
          }
#endif
        if (size>size_warning) Cerr << " OK" << finl;
        break;
      }
    case ArrOfDouble::TEMP_STORAGE:
      {
        // Allocation de la memoire sur un tas special.
        // La memoire ne sera pas rendue au systeme mais conservee pour une
        // reutilisation ulterieure.
        Memoire& memoire = Memoire::Instance();
        if (size>size_warning) Cerr << "Warning: Allocating or reusing a DoubleTrav of " << size << " elements ...";
        d_ptr_trav_ = memoire.add_trav_double(size);
        assert(d_ptr_trav_ != 0);
        data_ = d_ptr_trav_->d_ptr_();
        if (size>size_warning) Cerr << " OK" << finl;
        break;

      }
    default:
      Process::exit();
    }
  ref_count_ = 1;
  size_ = size;

  assert(data_ != 0);
}

// Description:
//  Detruit la zone de memoire allouee.
// Precondition:
//  ref_count == 0 (la zone de memoire ne doit etre referencee nulle part)
VDoubledata::~VDoubledata()
{
  assert(ref_count_ == 0);
  if (d_ptr_trav_ == 0)
    {
      // Stockage STANDARD
      delete[] data_;
    }
  else
    {
      // Stockage TEMP_STORAGE
      d_ptr_trav_->unlock();
      d_ptr_trav_ = 0;
    }
  data_ = 0;  // paranoia: si size_==-1 c'est qu'on pointe sur un zombie
  size_ = -1; //  (pointeur vers un objet qui a ete detruit)
}

// Description: renvoie ref_count_
inline int VDoubledata::ref_count() const
{
  return ref_count_;
}

// Description: renvoie size_
inline int VDoubledata::get_size() const
{
  return size_;
}

// Description:
//     Un nouveau tableau utilise cette zone memoire :
//     incremente ref_count
// Retour: int
//    Signification: ref_count
inline int VDoubledata::add_one_ref()
{
  return ++ref_count_;
}

// Description:
//     Un tableau de moins utilise cette zone memoire
//     decremente ref_count
// Precondition:
//     ref_count_ > 0
// Retour: int
//    Signification: ref_count
inline int VDoubledata::suppr_one_ref()
{
  assert(ref_count_ > 0);
  return (--ref_count_);
}

// Description: renvoie data_
inline double * VDoubledata::get_data()
{
  return data_;
}

// Description: renvoie data_
inline const double * VDoubledata::get_data() const
{
  return data_;
}

// Description: Constructeur par copie. Interdit : genere une erreur !
VDoubledata::VDoubledata(const VDoubledata& v)
{
  Cerr << "Error in VDoubledata: VDoubledata (const VDoubledata & v)" << finl;
  Process::exit();
}

// Description: Operateur= interdit. Genere une erreur !
VDoubledata& VDoubledata::operator=(const VDoubledata& v)
{
  Cerr << "Error in VDoubledata::operator=(const VDoubledata & v)" << finl;
  Process::exit();
  return *this;
}

// ******************************************************************
//
//             Implementation des methodes de ArrOfDouble
//
// ******************************************************************

// Description:
//    Ecriture du tableau sur "os" :
//    ecrit le nombre d'elements suivi des valeurs du tableau
// Precondition:
// Parametre: Sortie& os
//    Signification: le flot de sortie a utiliser
// Retour: Sortie&
//    Signification: le flot de sortie modifie
Sortie& ArrOfDouble::printOn(Sortie& os) const
{
  int sz=size_array();
  os << sz << finl;
  if (sz > 0)
    {
      const double* v = data_;
      os.put(v,sz,sz);
    }
  return os;
}

// Description:
//    Lecture d'un tableau dans un flot d'entree
//    Lit le nombre d'elements suivi des elements eux-memes
// Precondition:
//    Le tableau doit etre "resizable" (voir precondition de resize_array_)
// Parametre: Entree& is
//    Signification: le flot d'entree a utiliser
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: Entree&
//    Signification: le flot d'entree modifie
//    Contraintes:
// Exception:
//    Si la taille lue est negative, erreur.
Entree& ArrOfDouble::readOn(Entree& is)
{
  int sz;
  is >> sz;
  if (sz >= 0)
    {
      // Appel a la methode sans precondition sur le type derive
      // (car readOn est virtuelle, les autres proprietes seront initialisees correctement)
      resize_array_(sz);
      if (sz > 0)
        {
          double* v = data_;
          is.get(v,sz);
        }
    }
  else
    {
      Cerr << "Error in ArrOfDouble:readOn : size = " << sz << finl;
      assert(0);
      exit();
    }
  return is;
}

// Description:
//  Destructeur : appelle detach_array()
ArrOfDouble::~ArrOfDouble()
{
  detach_array();
  size_array_ = -1; // Paranoia: si size_array_==-1, c'est un zombie
}

// Description:
//  Constructeur par defaut: cree un tableau "detache",
//  soit p_==0, data_==0, size_array_==0, smart_resize_==0
ArrOfDouble::ArrOfDouble() :
  p_(0),
  data_(0),
  size_array_(0),
  memory_size_(0),
  smart_resize_(0),
  storage_type_(STANDARD)
{
}

// Description:
//  Cree un tableau de taille n avec allocation standard (voir set_mem_storage).
//  Le tableau est rempli avec zero par defaut. Pour creer un tableau non initialise, utiliser:
//   ArrOfDouble toto;
//   toto.resize_array(n, NOCOPY_NOINIT);
// Parametre: int n
//    Signification: taille du tableau
ArrOfDouble::ArrOfDouble(int n) :
  p_(0),
  data_(0),
  size_array_(n),
  memory_size_(n),
  smart_resize_(0),
  storage_type_(STANDARD)
{
  if (n)
    {
      p_ = new VDoubledata(n, STANDARD);
      data_ = p_->get_data();
      fill_default_value(COPY_INIT, 0, n);
    }
}

// Description:
//     Cree un tableau de taille n
//     toutes les cases sont initialisees a x
// Precondition:
// Parametre: int n
//    Signification: taille du tableau
// Parametre: double x
//    Signification: valeur pour initialiser le tableau
/*
ArrOfDouble::ArrOfDouble(int n, double x) :
  p_(0),
  data_(0),
  size_array_(n),
  memory_size_(n),
  smart_resize_(0),
  storage_type_(STANDARD)
{
  if (n) {
    p_ = new VDoubledata(n, STANDARD);
    data_ = p_->get_data();
    *this = x;
  }
}
*/
// Description:
//     Constructeur par copie. On alloue une nouvelle zone de memoire
//     et on copie le contenu du tableau (il s'agit d'un "deep copy").
//     L'attribut smart_resize_ est copie aussi.
//     Si le tableau A est de taille nulle, on cree un tableau "detache",
//     sinon on cree un tableau "normal".
// Parametre: const ArrOfDouble& A
//    Signification: le tableau a copier
ArrOfDouble::ArrOfDouble(const ArrOfDouble& A): Array_base()
{
  const int size = A.size_array();
  if (size > 0)
    {
      // Creation d'un tableau "normal"
      storage_type_ = STANDARD;
      p_ = new VDoubledata(size, STANDARD);
      data_ = p_->get_data();
      size_array_ = size;
      memory_size_ = size;
      smart_resize_ = A.smart_resize_;
      inject_array(A);
    }
  else
    {
      // Creation d'un tableau "detache"
      p_ = 0;
      data_ = 0;
      size_array_ = 0;
      memory_size_ = 0;
      smart_resize_ = 0;
      storage_type_ = STANDARD;
    }
}

// Description:
//   Change le mode d'allocation memoire lors des resize
//   (voir VDoubledata et Double_ptr_trav)
//   Exemple pour creer un tableau avec allocation temporaire:
//    DoubleTab tab; // Creation d'un tableau vide
//    tab.set_mem_storage(TEMP_STORAGE); // Changement de mode d'allocation
//    tab.resize(n); // Allocation memoire
void ArrOfDouble::set_mem_storage(const Storage storage)
{
  storage_type_ = storage;
}

// Description:
//   Renvoie le mode d'allocation du tableau (qui sera utilise
//   lors du prochain resize si changement de taille).
//   (voir VDoubledata et Double_ptr_trav)
enum ArrOfDouble::Storage ArrOfDouble::get_mem_storage() const
{
  return storage_type_;
}

// Description:
//   Change le mode l'allocation memoire: reallocation d'un tableau
//   a chaque changement de taille (flag = 0) ou reallocation
//   uniquement si la taille augmente et par doublement de la taille
//   du tableau (flag = 1).
void ArrOfDouble::set_smart_resize(int flag)
{
  assert(flag == 0 || flag == 1);
  smart_resize_ = flag;
}

// Description:
//    Remet le tableau dans l'etat obtenu avec le constructeur par defaut
//    (libere la memoire mais conserve le mode d'allocation memoire actuel)
void ArrOfDouble::reset()
{
  detach_array();
}

// Description:
//    Copie les donnees du tableau m.
//    Si "m" n'a pas la meme taille que "*this", on fait un resize_array.
//    Ensuite, on copie les valeurs de "m" dans "*this".
//    Le type de tableau (methode d'allocation) n'est pas copie.
// Precondition:
//    preconditions identiques a resize_array()
// Parametre: const ArrOfDouble& m
//    Signification: la tableau a copier
// Retour:  ArrOfDouble&
//    Signification: *this
ArrOfDouble& ArrOfDouble::operator=(const ArrOfDouble& m)
{
  if (&m != this)
    {
      const int new_size = m.size_array();
      // On utilise la methode resize_array() qui teste le type derive de l'objet
      // (resize interdit sur un type derive)
      resize_array(new_size, NOCOPY_NOINIT);
      inject_array(m);
    }
  return *this;
}
/*
// Description:
//   Affectation d'un tableau de type FArrOfDouble
//   (tableau stocke sur disque).
// Precondition:
//   Identiques a resize_array()
// Parametre: const FArrOfDouble& A
//    Signification: le tableau a copier
ArrOfDouble& ArrOfDouble::operator=(const FArrOfDouble& m)
{
  const int new_size = m.size_array();
  resize_array(new_size, NOCOPY_NOINIT);
  double * data = data_;
  for(int i=0; i<new_size; i++)
    data[i] = m[i];

  return *this;
}
*/

// Description: appelle operator=(a)
ArrOfDouble& ArrOfDouble::copy_array(const ArrOfDouble& a)
{
  operator=(a);
  return *this;
}

// Description: methode virtuelle identique a resize_array(), permet de traiter
//   de facon generique les ArrOf, Vect et Tab.
//   Si l'objet est de type ArrOfDouble, appel a resize_array(n)
// Prerequis: le tableau doit etre "resizable" (voir resize_array()).
//   S'il est d'un type derive (Vect ou Tab), il ne doit pas avoir de descripteur
//   parallele si la taille est effectivement modifiee.
void ArrOfDouble::resize_tab(int n, Array_base::Resize_Options opt)
{
  resize_array(n, opt);
}

// Description:
//  Si besoin, alloue une nouvelle zone de memoire,
//  copie les donnees et efface l'ancienne zone de memoire.
//  Attention, on suppose que cette methode est appelee par
//  resize_array().
//  Attention: si ref_count_>1, l'appel a resize_array() est
//  autorise uniquement si la nouvelle taille est identique
//  a la precedente (auquel cas on ne fait rien)
//  Si ref_count_ == 1, l'appel est invalide si p_->data_ est
//   different de data_ (le tableau a ete construit avec ref_array()
//   avec start > 0)
// Precondition:
//  Le tableau doit etre de type "detache" ou "normal" avec
//  ref_count==1, et il faut new_size >= 0
//  On suppose que size_array contient encore le nombre d'elements
//  valides avant changement de taille.
// Parametre: new_size
//  Signification: nouvelle taille demandee pour le tableau.
// Parametre: options
//  Signification: voir ArrOfDouble::Resize_Options
// Postcondition:
//  p_ et data_ sont mis a jour, mais pas size_array_ !!!
//  (on suppose que c'est fait dans resize_array()).
//  Si la nouvelle taille est nulle, on detache le tableau.
void  ArrOfDouble::memory_resize(int new_size, Array_base::Resize_Options opt)
{
  assert(new_size >= 0);

  // Si new_size==size_array_, on ne fait rien, c'est toujours autorise
  if (new_size == size_array_)
    return;

  // Sinon, il faut que le resize soit autorise, c'est a dire
  //  - soit le tableau est detache
  //  - soit le tableau est normal et il n'y a pas d'autre reference au tableau
  assert((data_ == 0) || ((p_ != 0) && (p_->get_data() == data_) && (ref_count() == 1)));

  // Occupation memoire de l'ancien tableau:
  int old_mem_size = memory_size_;
  assert((!p_) || (p_->get_size() == memory_size_));
  assert(memory_size_ >= size_array_);

  // Occupation memoire du nouveau tableau :
  // Si smart_resize, on prend au moins deux fois la taille
  // precedente, ou new_size
  int new_mem_size = new_size;
  if (smart_resize_)
    {
      if (new_size <= old_mem_size)
        new_mem_size = old_mem_size;
      else if (new_size < old_mem_size * 2)
        new_mem_size = old_mem_size * 2;
      else if (new_size > old_mem_size && old_mem_size > INT_MAX / 2)
        new_mem_size = INT_MAX;
    }

  if (new_mem_size != old_mem_size)
    {
      // detach_array() efface le contenu de size_array_. On le met de cote:
      const int old_size_array = size_array_;
      if (new_mem_size == 0)
        {
          // La nouvelle taille est nulle, on cree un tableau "detache"
          detach_array();
        }
      else
        {
          // Allocation d'une nouvelle zone
          VDoubledata * new_p = new VDoubledata(new_mem_size, storage_type_);
          double * new_data = new_p->get_data();
          // Raccourci si le tableau etait "detache", inutile de copier
          // les anciennes donnees. On copie si COPY_OLD est demande
          int copy_size = 0;
          if (data_ != 0)
            {
              // Calcul du nombre d'elements a copier vers la nouvelle
              // zone de memoire : c'est le min de l'ancienne et de
              // la nouvelle taille.
              if (opt != NOCOPY_NOINIT)
                {
                  copy_size = size_array_;
                  if (new_size < copy_size)
                    copy_size = new_size;
                  // Copie des valeurs dans le nouveau tableau
                  for (int i = 0; i < copy_size; i++)
                    new_data[i] = data_[i];
                }
              // Destruction de l'ancienne zone (si plus aucune reference)
              detach_array();
            }
          // On attache la nouvelle zone de memoire
          p_ = new_p;
          data_ = new_data;
          memory_size_ = new_mem_size;
          // Initialisation des cases supplementaires avec une valeur par defaut
          fill_default_value(opt, copy_size, new_mem_size - copy_size);
          // Restaure l'ancienne valeur de size_array_
          size_array_ = old_size_array;
        }
    }
  else
    {
      // Pas de reallocation, initialisation si besoin
      if (opt == COPY_INIT && new_size > size_array_)
        {
          fill_default_value(opt, size_array_, new_size - size_array_);
        }
    }
}

// Description:
//  Remplit "nb" cases consecutives du tableau a partir de la case "first"
//  avec une valeur par defaut.
//  Cette fonction est appelee lors d'un resize pour initialiser les
//  cases nouvellement creees.
//  Le comportement depend actuellement du type de tableau :
//  * Tableau de type "smart_resize":
//    * en mode debug (macro NDEBUG non definie) le tableau est initialise
//      avec une valeur invalide.
//    * en optimise, le tableau n'est pas initialise
//  * Tableau normal :
//    Le tableau est initialise avec la valeur 0. Ce comportement est choisi
//    pour des raisons de compatibilite avec l'implementation precedente.
//    Cette specification pourrait etre modifiee prochainement pour des raisons
//    de performances (pour ne pas avoir a initialiser inutilement les tableaux).
//    DONC: il faut supposer desormais que les nouvelles cases ne sont pas
//    initialisees lors d'un resize.
// Parametre: first
//  Signification: premiere case a initialiser.
//  Contrainte:    (nb==0) ou (0 <= first < memory_size_)
// Parametre: nb
//  Signification: nombre de cases a initialiser.
//  Contrainte:    (nb==0) ou (0 < nb <= memory_size_ - first)
void  ArrOfDouble::fill_default_value(Array_base::Resize_Options opt, int first, int nb)
{
  assert((nb == 0) || (first >= 0 && first < memory_size_));
  assert((nb == 0) || (nb > 0 && nb <= memory_size_ - first));
  double * data = data_;
  assert(data!=0 || nb==0);
  data += first;
  if (opt != COPY_INIT)
    {
      // On initialise uniquement en mode debug
#ifndef NDEBUG
      // B.M. L'utilisation de NAN n'est pas possible car elle provoque une erreur
      // dans double & operator[] a cause de assert(...DMAXFLOAT). Du coup il est
      // impossible de mettre une valeur dans le tableau autrement qu'avec append_array()
#if 0
      // Ceci represente un NAN. N'importe quelle operation avec ca fait encore un NAN.
      // Si c'est pas portable, on peut remplacer par DMAX_FLOAT sur les autres machines.
      // Attention, il faut long long car sinon warning sur gcc 4.1 et i686
      static const unsigned long long VALEUR_INVALIDE =
        0x7ff7ffffffffffffULL;
      /* static const double VALEUR_INVALIDE = DMAXFLOAT;    */

      // On utilise "memcpy" et non "=" car "=" peut provoquer une exception
      // si la copie passe par le fpu.
      for (int i = 0; i < nb; i++)
        memcpy(data + i, & VALEUR_INVALIDE, sizeof(double));
#else
      // Alternative acceptable... ?
      for (int i = 0; i < nb; i++)
        data[i] = DMAXFLOAT*0.999;
#endif
#endif
    }
  else
    {
      // Comportement pour les tableaux normaux : compatibilite avec la
      // version precedente : on initialise avec 0.
      for (int i = 0; i < nb; i++)
        data[i] = (double) 0;
    }
}

// ****************************************************************
//
//         Fonctions non membres de la classe ArrOfDouble
//
// ****************************************************************

// Description:
//  Renvoie 1 si les tableaux "v" et "a" sont de la meme taille
//  et contiennent les memes valeurs au sens strict, sinon renvoie 0.
//  Le test est !(v[i]!=a[i])
int operator==(const ArrOfDouble& v, const ArrOfDouble& a)
{
  const int n = v.size_array();
  const int na = a.size_array();
  int resu = 1;
  if (n != na)
    {
      resu = 0;
    }
  else
    {
      const double* vv = v.addr();
      const double* av = a.addr();
      int i;
      for (i = 0; i < n; i++)
        {
          if (av[i] != vv[i])
            {
              resu = 0;
              break;
            }
        }
    }
  return resu;
}

// Description:
//    Retourne l'indice du min ou -1 si le tableau est vide
// Precondition:
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
// Retour: int
//    Signification: indice du min
int imin_array(const ArrOfDouble& dx)
{
  int indice_min = -1;
  const int size = dx.size_array();
  if (size > 0)
    {
      indice_min = 0;
      double valeur_min = dx[0];
      for(int i = 1; i < size; i++)
        {
          const double val = dx[i];
          if(val < valeur_min)
            {
              indice_min = i;
              valeur_min = val;
            }
        }
    }
  return indice_min;
}

// Description:
//    Retourne l'indice du max ou -1 si le tableau est vide
// Precondition:
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
// Retour: int
//    Signification: indice du max
int imax_array(const ArrOfDouble& dx)
{
  int indice_max = -1;
  const int size = dx.size_array();
  if (size > 0)
    {
      indice_max = 0;
      double valeur_max = dx[0];
      for(int i = 1; i < size; i++)
        {
          const double val = dx[i];
          if(val > valeur_max)
            {
              indice_max = i;
              valeur_max = val;
            }
        }
    }
  return indice_max;
}

// Description:
//    Retourne la valeur minimale
// Precondition:
//    Le tableau doit contenir au moins une valeur
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
// Retour: double
//    Signification: valeur du min
double min_array(const ArrOfDouble& dx)
{
  const int size = dx.size_array();
  assert(size > 0);
  double valeur_min = dx[0];
  for(int i = 1; i < size; i++)
    {
      const double val = dx[i];
      if (val < valeur_min)
        valeur_min = val;
    }
  return valeur_min;
}

// Description:
//    Retourne la valeur maximale
// Precondition:
//    Le tableau doit contenir au moins une valeur
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
// Retour: double
//    Signification: valeur du max
double max_array(const ArrOfDouble& dx)
{
  const int size = dx.size_array();
  assert(size > 0);
  double valeur_max = dx[0];
  for(int i = 1; i < size; i++)
    {
      const double val = dx[i];
      if (val > valeur_max)
        valeur_max = val;
    }
  return valeur_max;
}

static inline double scalar_abs(double x)
{
  return dabs(x);
}

// Description:
//    Retourne le max des abs(i)
// Precondition:
//    Le tableau doit contenir au moins une valeur
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: double
//    Signification: valeur du max des valeurs absolues
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
double max_abs_array(const ArrOfDouble& dx)
{
  const int size = dx.size_array();
  assert(size > 0);
  double valeur_max = scalar_abs(dx[0]);
  for(int i = 1; i < size; i++)
    {
      const double val = scalar_abs(dx[i]);
      if (val > valeur_max)
        valeur_max = val;
    }
  return valeur_max;
}

// Description:
//    Retourne le min des abs(i)
// Precondition:
//    Le tableau doit contenir au moins une valeur
// Parametre: const ArrOfDouble& dx
//    Signification: tableau a utiliser
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: double
//    Signification: valeur du min des valeurs absolues
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
double min_abs_array(const ArrOfDouble& dx)
{
  const int size = dx.size_array();
  assert(size > 0);
  double v = scalar_abs(dx[0]);
  for(int i = 1; i < size; i++)
    {
      const double val = scalar_abs(dx[i]);
      if (val < v)
        v = val;
    }
  return v;
}

// Description:
//   Fonction de comparaison utilisee pour trier le tableau
//   dans ArrOfDouble::trier(). Voir man qsort
static True_int  fonction_compare_arrofdouble_ordonner(const void * data1, const void * data2)
{
  const double x = *(const double*)data1;
  const double y = *(const double*)data2;
  if (x < y)
    return -1;
  else if (x > y)
    return 1;
  else
    return 0;
}

// Description:
//   Tri des valeurs du tableau dans l'ordre croissant.
//   La fonction utilisee est qsort de stdlib (elle est en n*log(n)).
void ArrOfDouble::ordonne_array()
{
  const int size = size_array_;
  if (size > 1)
    {
      double * data = data_;
      qsort(data, size, sizeof(double),
            fonction_compare_arrofdouble_ordonner);
    }
}

// Description:
//   Tri des valeurs du tableau dans l'ordre croissant et suppresion des doublons
//   La fonction utilisee est qsort de stdlib (elle est en n*log(n)).
void ArrOfDouble::array_trier_retirer_doublons()
{
  const int size_ = size_array_;
  if (size_ <= 0)
    return;
  // Tri dans l'ordre croissant
  ordonne_array();
  // Retire les doublons (codage optimise en pointeurs)
  double last_value = *data_;
  double *src = data_ + 1;
  double *dest = data_ + 1;
  for (int i = size_ - 1; i != 0; i--)
    {
      double x = *(src++);
      if (x != last_value)
        {
          *(dest++) = x;
          last_value = x;
        }
    }
  int new_size_ = dest - data_;
  resize_array(new_size_);
}

// Description:
//    Fait pointer le tableau vers les memes donnees qu'un tableau
//    existant. Le tableau sera du meme type que le tableau m ("detache",
//    "normal"). Le tableau m ne doit pas etre de type "ref_data"
//    Attention, le tableau source et *this sont ensuite figes (resize_array()
//    interdit).
//   Attention: methode virtuelle: dans les classes derivee, cette methode
//    initialise les structures pour creer un tableau sequentiel.
// Parametre: const ArrOfDouble& m
//    Signification: le tableau a referencer (pas de type "ref_data"
//                   et different de *this !!!)
// Retour: ArrOfDouble&
//    Signification: *this
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
void ArrOfDouble::ref_array(ArrOfDouble& m, int start, int size)
{
  assert(&m != this);
  // La condition 'm n'est pas de type "ref_data"' est necessaire pour
  // attach_array().
  detach_array();
  attach_array(m, start, size);
}

// Description:
//    Fait pointer le tableau vers la zone de memoire "data_".
//    On detache la zone de memoire existante. Le tableau devient
//    de type "ref_data". Attention : ptr doit etre non nul.
//    La taille est initialisee avec size.
//   Attention: methode virtuelle: dans les classes derivee, cette methode
//    initialise les structures pour creer un tableau sequentiel.
//    Pour faire une ref sur un tableau parallele utiliser DoubleVect::ref()
// Parametre: double*
//    Signification: le tableau a recuperer. Si pointeur nul alors size
//                   doit etre nulle aussi et le tableau reste detache
// Parametre: int size
//    Signification: le nombre d'elements du tableau.
// Retour: ArrOfDouble&
//    Signification: *this
void ArrOfDouble::ref_data(double* ptr, int size)
{
  assert(ptr != 0 || size == 0);
  assert(size >= 0);
  detach_array();
  data_ = ptr;
  size_array_ = size;
  memory_size_ = size; // Pour passer les tests si on resize a la meme taille
}

// Description:
//  Amene le tableau dans l'etat "detache". C'est a dire:
//  Si le tableau est "detache" :
//   * ne rien faire
//  Si le tableau est "normal" :
//   * decremente le nombre de references a *p
//   * detruit *p si p->ref_count==0
//   * annule p_, data_ et size_array_
//  Si le tableau est "ref_data" :
//   * annule data_ et size_array_
// Retour: int
//    Signification: 1 si les donnees du tableau ont ete supprimees
// Precondition:
// Postcondition:
//  On a p_==0, data_==0 et size_array_==0, memory_size_ = 0
//  L'attribut smart_resize_ est conserve.
int ArrOfDouble::detach_array()
{
  int retour = 0;
  if (p_)
    {
      // Le tableau est de type "normal"
      // Si la zone de memoire n'est plus utilisee par personne,
      // on la detruit.
      if ((p_->suppr_one_ref()) == 0)
        {
          delete p_;
          retour = 1;
        }
      p_ = 0;
    }
  data_ = 0;
  size_array_ = 0;
  memory_size_ = 0;
  return retour;
}

// Description:
//    Amene le tableau dans l'etat "normal", "detache" ou "ref_array"
//    en associant une sous-zone de memoire du tableau m, definie par start et size
//    Si size < 0, on prend le tableau m jusqu'a la fin.
// Precondition:
//    Le tableau doit etre "detache"
// Parametre: const ArrOfDouble& m
//    Signification: tableau a utiliser
//                   le tableau doit etre different de *this !!!
// Retour:
//    Signification:
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
//    Si m est detache, le tableau reste detache,
//    si m est "ref_array", le tableau devient "ref_array",
//    sinon le tableau est "normal", avec ref_count > 1
//    Si m est de taille nulle, le tableau reste detache + Warning dans fichier .log
void ArrOfDouble::attach_array(const ArrOfDouble& m, int start, int size)
{
  // Le tableau doit etre detache
  assert(data_ == 0 && p_ == 0);
  // Le tableau doit etre different de *this
  assert(&m != this);
  if (size < 0)
    size = m.size_array_ - start;
  assert(start >= 0 && size >=0 && start + size <= m.size_array_);
  if (m.size_array() > 0)
    {
      p_ = m.p_;
      if (p_)
        p_->add_one_ref();
      data_ = m.data_ + start;
      size_array_ = size;
      memory_size_ = m.memory_size_ - start;
      smart_resize_ = m.smart_resize_;
    }
  else
    {
      // Cas particulier ou on attache un tableau de taille nulle:
      //  en theorie, c'est pareil qu'un tableau de taille non nulle, MAIS
      //  dans les operateurs (ex:Op_Dift_VDF_Face_Axi), une ref est construite
      //  avant que le tableau ne prenne sa taille definitive. Donc, pour ne pas
      //  empecher le resize, il ne faut pas attacher le tableau s'il n'a pas
      //  encore la bonne taille. Solution propre: reecrire les operateurs pour
      //  qu'ils ne prennent pas une ref avant que le tableau ne soit valide
      //  et faire p_ = m.p_ dans tous les cas.
      // Process::Journal() << "Warning ArrOfDouble::attach_array(m), m.size_array()==0, on n attache pas le tableau" << finl;
    }
}

// Description:
//    Copie les elements source[first_element_source + i]
//    dans les elements  (*this)[first_element_dest + i] pour 0 <= i < nb_elements
//    Les autres elements de (*this) sont inchanges.
// Precondition:
// Parametre:       const ArrOfDouble& m
//  Signification:   le tableau a utiliser, doit etre different de *this !
// Parametre:       int nb_elements
//  Signification:   nombre d'elements a copier, nb_elements >= -1.
//                   Si nb_elements==-1, on copie tout le tableau m.
//  Valeurs par defaut: -1
// Parametre:       int first_element_dest
//  Valeurs par defaut: 0
// Parametre:       int first_element_source
//  Valeurs par defaut: 0
// Retour: ArrOfDouble&
//    Signification: *this
//    Contraintes:
// Exception:
//    Sort en erreur si la taille du tableau m est plus grande que la
//    taille de tableau this.
// Effets de bord:
// Postcondition:
ArrOfDouble& ArrOfDouble::inject_array(const ArrOfDouble& source,
                                       int nb_elements,
                                       int first_element_dest,
                                       int first_element_source)
{
  assert(&source != this);
  assert(nb_elements >= -1);
  assert(first_element_dest >= 0);
  assert(first_element_source >= 0);

  if (nb_elements < 0)
    nb_elements = source.size_array();

  assert(first_element_source + nb_elements <= source.size_array());
  assert(first_element_dest + nb_elements <= size_array());

  if (nb_elements > 0)
    {
      double * addr_dest = data_ + first_element_dest;
      const double * addr_source = source.addr() + first_element_source;
      // PL: On utilise le memcpy car c'est VRAIMENT plus rapide (10% +vite sur RNR_G20)
      memcpy(addr_dest , addr_source, nb_elements * sizeof(double));
      /*
        int i;
        for (i = 0; i < nb_elements; i++) {
        addr_dest[i] = addr_source[i];
        } */
    }
  return *this;
}

// Description:
//    Retourne le nombre de references des donnees du tableau
//    si le tableau est "normal", -1 s'il est "detache" ou "ref_data"
// Retour: int
//    Signification: ref_count_
int ArrOfDouble::ref_count() const
{
  if (p_)
    return p_->ref_count();
  else
    return -1;
}

// Description:
//    Addition case a case sur toutes les cases du tableau
// Precondition:
//    la taille de y doit etre au moins egale a la taille de this
// Parametre: const ArrOfDouble& y
//    Signification: tableau a ajouter
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: ArrOfDouble&
//    Signification: *this
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
ArrOfDouble& ArrOfDouble::operator+=(const ArrOfDouble& y)
{
  assert(size_array()==y.size_array());
  double* dx = data_;
  const double* dy = y.data_;
  const int n = size_array();
  for (int i=0; i<n; i++)
    dx[i] += dy[i];
  return *this;
}

// Description:
//     ajoute la meme valeur a toutes les cases du tableau
// Precondition:
// Parametre: const double dy
//    Signification: valeur a ajouter
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: ArrOfDouble
//    Signification: *this
//    Contraintes:
// Exception:
// Effets de bord:
// Postcondition:
ArrOfDouble& ArrOfDouble::operator+=(const double dy)
{
  double * data = data_;
  const int n = size_array();
  for(int i=0; i < n; i++)
    data[i] += dy;
  return *this;
}
// Description:
//    Soustraction case a case sur toutes les cases du tableau
// Parametre: const ArrOfDouble& y
//    Signification: tableau de meme taille que *this
// Retour: ArrOfDouble&
//    Signification: *this
ArrOfDouble& ArrOfDouble::operator-=(const ArrOfDouble& y)
{
  const int size = size_array();
  assert(size == y.size_array());
  double * data = data_;
  const double * data_y = y.data_;
  for (int i=0; i < size; i++)
    data[i] -= data_y[i];
  return *this;
}


// Description:
//     soustrait la meme valeur a toutes les cases
// Retour: ArrOfDouble &
//    Signification: *this
ArrOfDouble& ArrOfDouble::operator-=(const double dy)
{
  double * data = data_;
  const int n = size_array();
  for(int i=0; i < n; i++)
    data[i] -= dy;
  return *this;
}


// Description:
//     muliplie toutes les cases par dy
// Retour: ArrOfDouble &
//    Signification: *this
ArrOfDouble& ArrOfDouble::operator*= (const double dy)
{
  double * data = data_;
  const int n = size_array();
  for(int i=0; i < n; i++)
    data[i] *= dy;
  return *this;
}


// Description:
//     divise toutes les cases par dy
// Retour: ArrOfDouble &
//    Signification: *this
ArrOfDouble& ArrOfDouble::operator/= (const double dy)
{
  const double i_dy = 1. / dy;
  operator*=(i_dy);
  return *this;
}

// Description:
//    Produit scalaire de deux "array".
//    (dotproduct_array remplace operator* car le nom indique clairement
//     que l'on fait un produit scalaire non distribue)
// Precondition:
// Parametre: const ArrOfDouble& dx
//    Signification: premier tableau
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Parametre: const ArrOfDouble& dy
//    Signification: second tableau (obligatoirement de meme taille)
//    Valeurs par defaut:
//    Contraintes:
//    Acces:
// Retour: double
//    Signification: produit scalaire
//    Contraintes:
double dotproduct_array (const ArrOfDouble& dx, const ArrOfDouble& dy)
{
  const int size = dx.size_array();
  assert(size == dy.size_array());
  double resultat = 0.;
  if (size > 0)
    {
      integer n = size;
      integer incx = 1;
      resultat = F77NAME(DDOT)(&n, dx.addr(), &incx, dy.addr(), &incx);
    }
  return resultat;
}

// Description:
//   Calcul de la norme du vecteur dx (fonction blas DNRM2,
//   soit racine carree(somme des dx[i]*dx[i]).
double norme_array(const ArrOfDouble& dx)
{
  integer n = dx.size_array();
  double resultat = 0.;
  if (n > 0)
    {
      integer incx = 1;
      resultat = F77NAME(DNRM2)(&n, &dx(0), &incx);
    }
  return resultat;
}

