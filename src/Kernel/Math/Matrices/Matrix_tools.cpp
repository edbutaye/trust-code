/****************************************************************************
* Copyright (c) 2015, CEA
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
// File:        Matrix_tools.cpp
// Directory:   $TRUST_ROOT/src/Kernel/Math/Matrices
// Version:     /main/2
//
//////////////////////////////////////////////////////////////////////////////

#include <Matrix_tools.h>
#include <Matrice.h>
#include <Matrice_Base.h>
#include <Matrice_Nulle.h>
#include <Matrice_Diagonale.h>
#include <Matrice_Morse.h>
#include <Matrice_Morse_Sym.h>
#include <IntTab.h>
#include <ArrOfDouble.h>
#include <Array_tools.h>

// conversion to morse format
void Matrix_tools::convert_to_morse_matrix( const Matrice_Base& in,
                                            Matrice_Morse&      out )
{
  IntTab stencil;
  ArrOfDouble coefficients;

  in.get_stencil_and_coefficients( stencil, coefficients );

  Matrix_tools::build_morse_matrix( in.nb_lignes( ),
                                    in.nb_colonnes( ),
                                    stencil,
                                    coefficients,
                                    out );
}



// conversion to symetric morse format
void Matrix_tools::convert_to_symmetric_morse_matrix( const Matrice_Base& in,
                                                      Matrice_Morse_Sym&  out )
{
  assert( in.nb_lignes( ) == in.nb_colonnes( ) );

  IntTab stencil;
  ArrOfDouble coefficients;

  in.get_symmetric_stencil_and_coefficients( stencil, coefficients );

  Matrix_tools::build_symmetric_morse_matrix( in.nb_lignes( ),
                                              stencil,
                                              coefficients,
                                              out );
}

// checking stencil
bool Matrix_tools::is_normalized_stencil( const IntTab& stencil )
{
  const int size = stencil.dimension( 0 );
  for ( int i=1; i<size; ++i )
    {
      int delta1 = stencil( i-1, 0 ) - stencil( i, 0 );
      int delta2 = stencil( i-1, 1 ) - stencil( i, 1 );
      int delta  = ( delta1 == 0 ) ? delta2 : delta1;

      if ( delta >= 0 )
        {
          return false;
        }
    }
  return true;
}

// checking symmetric stencil
bool Matrix_tools::is_normalized_symmetric_stencil( const IntTab& stencil )
{
  if ( ! ( is_normalized_stencil( stencil ) ) )
    {
      Cerr << "non-normalized stencil" << finl;
      return false;
    }

  const int size = stencil.dimension( 0 );
  for ( int i=0; i<size; ++i )
    {
      int delta = stencil( i, 0 ) - stencil( i, 1 );
      if ( delta > 0 )
        {
          Cerr << "( " << i << " ) line index > column index : ( " << stencil( i, 0 ) << ", " << stencil( i, 1 ) << " )" << finl;
          return false;
        }
    }

  return true;
}


// building morse matrices
void Matrix_tools::allocate_morse_matrix( const int&  nb_lines,
                                          const int&  nb_columns,
                                          const IntTab&  stencil,
                                          Matrice_Morse& matrix )
{
  assert( is_normalized_stencil( stencil ) );

  const int nb_coefficients = stencil.dimension( 0 );

  matrix.dimensionner( nb_lines,
                       nb_columns,
                       nb_coefficients );

  if ( nb_coefficients > 0 )
    {
      matrix.tab1_ = 0;
      matrix.tab1_( 0 ) = 1;
      for ( int i=0; i<nb_coefficients; ++i )
        {
          assert( stencil( i ,0 ) >= 0         );
          assert( stencil( i ,0 ) < nb_lines   );
          assert( stencil( i ,1 ) >= 0         );
          assert( stencil( i ,1 ) < nb_columns );

          matrix.tab1_( stencil( i, 0 ) + 1 ) += 1;
          matrix.tab2_( i ) = stencil( i, 1 ) + 1;
        }
      for ( int i=0; i<nb_lines; ++i )
        {
          matrix.tab1_( i + 1 ) += matrix.tab1_( i );
        }
    }
}

void Matrix_tools::build_morse_matrix( const int&      nb_lines,
                                       const int&      nb_columns,
                                       const IntTab&      stencil,
                                       const ArrOfDouble& coefficients,
                                       Matrice_Morse&     matrix )
{
  assert( is_normalized_stencil( stencil ) );

  const int nb_coefficients = stencil.dimension( 0 );
  assert( nb_coefficients == coefficients.size_array( ) );

  matrix.dimensionner( nb_lines,
                       nb_columns,
                       nb_coefficients );

  if ( nb_coefficients > 0 )
    {
      matrix.tab1_ = 0;
      matrix.tab1_( 0 ) = 1;
      for ( int i=0; i<nb_coefficients; ++i )
        {
          assert( stencil( i ,0 ) >= 0         );
          assert( stencil( i ,0 ) < nb_lines   );
          assert( stencil( i ,1 ) >= 0         );
          assert( stencil( i ,1 ) < nb_columns );

          matrix.tab1_( stencil( i, 0 ) + 1 ) += 1;
          matrix.tab2_( i ) = stencil( i, 1 ) + 1;
          matrix.coeff_( i ) = coefficients[ i ];
        }
      for ( int i=0; i<nb_lines; ++i )
        {
          matrix.tab1_( i + 1 ) += matrix.tab1_( i );
        }
    }
}



// building symmetric morse matrices
void Matrix_tools::allocate_symmetric_morse_matrix( const int&      order,
                                                    const IntTab&      stencil,
                                                    Matrice_Morse_Sym& matrix )
{
  assert( is_normalized_symmetric_stencil( stencil ) );

  const int nb_coefficients = stencil.dimension( 0 );

  matrix.dimensionner( order,
                       order,
                       nb_coefficients );

  if ( nb_coefficients > 0 )
    {
      matrix.tab1_ = 0;
      matrix.tab1_( 0 ) = 1;
      for ( int i=0; i<nb_coefficients; ++i )
        {
          assert( stencil( i ,0 ) >= 0               );
          assert( stencil( i ,0 ) < order            );
          assert( stencil( i ,1 ) >= 0               );
          assert( stencil( i ,1 ) < order            );
          assert( stencil( i, 0 ) <= stencil( i, 1 ) );

          matrix.tab1_( stencil( i, 0 ) + 1 ) += 1;
          matrix.tab2_( i ) = stencil( i, 1 ) + 1;
        }
      for ( int i=0; i<order; ++i )
        {
          matrix.tab1_( i + 1 ) += matrix.tab1_( i );
        }
    }
  matrix.symetrique_ = 1;
}


void Matrix_tools::build_symmetric_morse_matrix( const int&      order,
                                                 const IntTab&      stencil,
                                                 const ArrOfDouble& coefficients,
                                                 Matrice_Morse_Sym& matrix )
{
  assert( is_normalized_symmetric_stencil( stencil ) );

  const int nb_coefficients = stencil.dimension( 0 );
  assert( nb_coefficients == coefficients.size_array( ) );

  matrix.dimensionner( order,
                       order,
                       nb_coefficients );

  if ( nb_coefficients > 0 )
    {
      matrix.tab1_ = 0;
      matrix.tab1_( 0 ) = 1;
      for ( int i=0; i<nb_coefficients; ++i )
        {
          assert( stencil( i ,0 ) >= 0               );
          assert( stencil( i ,0 ) < order            );
          assert( stencil( i ,1 ) >= 0               );
          assert( stencil( i ,1 ) < order            );
          assert( stencil( i, 0 ) <= stencil( i, 1 ) );

          matrix.tab1_( stencil( i, 0 ) + 1 ) += 1;
          matrix.tab2_( i )  = stencil( i, 1 ) + 1;
          matrix.coeff_( i ) = coefficients[ i ];
        }
      for ( int i=0; i<order; ++i )
        {
          matrix.tab1_( i + 1 ) += matrix.tab1_( i );
        }
    }
  matrix.symetrique_ = 1;
}


// allocation for scaled addition
void Matrix_tools::allocate_for_scaled_addition( const Matrice& A,
                                                 const Matrice& B,
                                                 Matrice&       C )
{
  assert( A.valeur( ).nb_lignes( )   == B.valeur( ).nb_lignes( )   );
  assert( A.valeur( ).nb_colonnes( ) == B.valeur( ).nb_colonnes( ) );

  IntTab A_stencil;
  A.valeur( ).get_stencil( A_stencil );
  const int A_size = A_stencil.dimension( 0 );

  IntTab B_stencil;
  B.valeur( ).get_stencil( B_stencil );
  const int B_size = B_stencil.dimension( 0 );

  int size = A_size + B_size;
  IntTab stencil;
  stencil.set_smart_resize( 1 );
  stencil.resize( size, 2 );
  stencil.resize( 0, 2 );

  for ( int i=0; i<A_size; ++i )
    {
      stencil.append_line( A_stencil( i, 0 ), A_stencil( i, 1 ) );
    }

  for ( int i=0; i<B_size; ++i )
    {
      stencil.append_line( B_stencil( i, 0 ), B_stencil( i, 1 ) );
    }

  tableau_trier_retirer_doublons( stencil );

  C.typer( "Matrice_Morse" );
  Matrice_Morse& C_ = ref_cast( Matrice_Morse, C.valeur( ) );

  allocate_morse_matrix( A.valeur( ).nb_lignes( ),
                         A.valeur( ).nb_colonnes( ),
                         stencil,
                         C_ );
}

void Matrix_tools::allocate_for_symmetric_scaled_addition( const Matrice& A,
                                                           const Matrice& B,
                                                           Matrice&       C )
{
  assert( A.valeur( ).nb_lignes( )   == A.valeur( ).nb_lignes( )   );
  assert( A.valeur( ).nb_lignes( )   == B.valeur( ).nb_lignes( )   );
  assert( A.valeur( ).nb_colonnes( ) == B.valeur( ).nb_colonnes( ) );

  IntTab A_stencil;
  A.valeur( ).get_symmetric_stencil( A_stencil );
  const int A_size = A_stencil.dimension( 0 );

  IntTab B_stencil;
  B.valeur( ).get_symmetric_stencil( B_stencil );
  const int B_size = B_stencil.dimension( 0 );

  int size = A_size + B_size;
  IntTab stencil;
  stencil.set_smart_resize( 1 );
  stencil.resize( size, 2 );
  stencil.resize( 0, 2 );

  for ( int i=0; i<A_size; ++i )
    {
      stencil.append_line( A_stencil( i, 0 ), A_stencil( i, 1 ) );
    }

  for ( int i=0; i<B_size; ++i )
    {
      stencil.append_line( B_stencil( i, 0 ), B_stencil( i, 1 ) );
    }

  tableau_trier_retirer_doublons( stencil );

  C.typer( "Matrice_Morse_Sym" );
  Matrice_Morse_Sym& C_ = ref_cast( Matrice_Morse_Sym, C.valeur( ) );

  allocate_symmetric_morse_matrix( A.valeur( ).nb_lignes( ),
                                   stencil,
                                   C_ );
}


// scaled addition
void Matrix_tools::add_scaled_matrices( const Matrice& A,
                                        const double&  alpha,
                                        const Matrice& B,
                                        const double&  beta,
                                        Matrice&       C )
{
  assert( sub_type( Matrice_Morse, C.valeur( ) ) );
  Matrice_Morse& C_ = ref_cast( Matrice_Morse, C.valeur( ) );

  assert( C_.check_sorted_morse_matrix_structure( ) );

  IntTab A_stencil;
  ArrOfDouble A_coefficients;
  A.valeur( ).get_stencil_and_coefficients( A_stencil,
                                            A_coefficients );

  IntTab B_stencil;
  ArrOfDouble B_coefficients;
  B.valeur( ).get_stencil_and_coefficients( B_stencil,
                                            B_coefficients );

  int A_k = 0;
  int B_k = 0;

  C_.coeff_ = 0.0;

  const int nb_lines = C_.nb_lignes( );
  for ( int i=0; i<nb_lines; ++i )
    {
      int k0   = C_.tab1_( i ) - 1;
      int k1   = C_.tab1_( i + 1 ) - 1;

      for ( int k=k0; k<k1; ++k )
        {
          int j = C_.tab2_( k ) - 1;

          if ( ( A_k < A_stencil.dimension( 0 ) ) && ( A_stencil( A_k, 0 ) == i ) && ( A_stencil( A_k, 1 ) == j ) )
            {
              C_.coeff( k ) += alpha * A_coefficients[ A_k ];
              ++A_k;
            }

          if ( ( B_k < B_stencil.dimension( 0 ) ) && ( B_stencil( B_k, 0 ) == i ) && ( B_stencil( B_k, 1 ) == j ) )
            {
              C_.coeff( k ) += beta * B_coefficients[ B_k ];
              ++B_k;
            }
        }
    }

  assert( A_k == A_coefficients.size_array( ) );
  assert( B_k == B_coefficients.size_array( ) );
}

void Matrix_tools::add_symmetric_scaled_matrices( const Matrice& A,
                                                  const double&  alpha,
                                                  const Matrice& B,
                                                  const double&  beta,
                                                  Matrice&       C )
{
  assert( sub_type( Matrice_Morse_Sym, C.valeur( ) ) );
  Matrice_Morse_Sym& C_ = ref_cast( Matrice_Morse_Sym, C.valeur( ) );

  assert( C_.check_sorted_symmetric_morse_matrix_structure( ) );

  IntTab A_stencil;
  ArrOfDouble A_coefficients;
  A.valeur( ).get_symmetric_stencil_and_coefficients( A_stencil,
                                                      A_coefficients );

  IntTab B_stencil;
  ArrOfDouble B_coefficients;
  B.valeur( ).get_symmetric_stencil_and_coefficients( B_stencil,
                                                      B_coefficients );

  int A_k = 0;
  int B_k = 0;

  C_.coeff_ = 0.0;

  const int nb_lines = C_.nb_lignes( );
  for ( int i=0; i<nb_lines; ++i )
    {
      int k0   = C_.tab1_( i ) - 1;
      int k1   = C_.tab1_( i + 1 ) - 1;

      for ( int k=k0; k<k1; ++k )
        {
          int j = C_.tab2_( k ) - 1;

          if ( ( A_k < A_stencil.dimension( 0 ) ) && ( A_stencil( A_k, 0 ) == i ) && ( A_stencil( A_k, 1 ) == j ) )
            {
              C_.coeff( k ) += alpha * A_coefficients[ A_k ];
              ++A_k;
            }

          if ( ( B_k < B_stencil.dimension( 0 ) ) && ( B_stencil( B_k, 0 ) == i ) && ( B_stencil( B_k, 1 ) == j ) )
            {
              C_.coeff( k ) += beta * B_coefficients[ B_k ];
              ++B_k;
            }
        }
    }

  assert( A_k == A_coefficients.size_array( ) );
  assert( B_k == B_coefficients.size_array( ) );
}

// stencil analysis
bool Matrix_tools::is_null_stencil( const IntTab& stencil )
{
  return ( stencil.dimension( 0 ) == 0 );
}

bool Matrix_tools::is_diagonal_stencil( const int& nb_lines,
                                        const int& nb_columns,
                                        const IntTab& stencil )
{
  if ( nb_lines != nb_columns )
    {
      return false;
    }

  const int size = stencil.dimension( 0 );

  if ( size == 0 )
    {
      return false;
    }

  for ( int i=0; i<size; ++i )
    {
      if ( stencil( i, 0 ) != stencil( i, 1 ) )
        {
          return false;
        }

      if ( stencil( i, 0 ) >= nb_lines )
        {
          return false;
        }

      if ( stencil( i, 1 ) >= nb_columns )
        {
          return false;
        }
    }

  return true;
}

void Matrix_tools::allocate_from_stencil( const int& nb_lines,
                                          const int& nb_columns,
                                          const IntTab& stencil,
                                          Matrice&      matrix )
{
  if ( is_null_stencil( stencil ) )
    {
      matrix.typer( "Matrice_Nulle" );
      Matrice_Nulle& matrix_ = ref_cast( Matrice_Nulle, matrix.valeur( ) );
      matrix_.dimensionner( nb_lines, nb_columns );
    }
  else if ( is_diagonal_stencil( nb_lines,
                                 nb_columns,
                                 stencil ) )
    {
      matrix.typer( "Matrice_Diagonale" );
      Matrice_Diagonale& matrix_ = ref_cast( Matrice_Diagonale, matrix.valeur( ) );
      matrix_.dimensionner( nb_lines );
    }
  else
    {
      matrix.typer( "Matrice_Morse" );
      Matrice_Morse& matrix_ = ref_cast( Matrice_Morse, matrix.valeur( ) );
      allocate_morse_matrix( nb_lines,
                             nb_columns,
                             stencil,
                             matrix_ );
    }
}

// extending a matrix's stencil
void Matrix_tools::extend_matrix_stencil( const IntTab& stencil,
                                          Matrice&      matrix )
{
  if ( ! ( is_null_stencil( stencil ) ) )
    {
      const int nb_lines   = matrix.valeur( ).nb_lignes( );
      const int nb_columns = matrix.valeur( ).nb_colonnes( );

      IntTab full_stencil;
      matrix.valeur( ).get_stencil( full_stencil );
      full_stencil.set_smart_resize( 1 );
      const int size = stencil.dimension( 0 );
      for ( int i=0; i<size; ++i )
        {
          full_stencil.append_line( stencil( i, 0 ), stencil( i, 1 ) ) ;
        }

      tableau_trier_retirer_doublons( full_stencil );

      allocate_from_stencil( nb_lines,
                             nb_columns,
                             full_stencil,
                             matrix );
    }
}
