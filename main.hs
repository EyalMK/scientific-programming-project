{-# LANGUAGE FlexibleContexts #-}
import Numeric.LinearAlgebra
import Data.List (maximumBy, sort)
import Data.Ord (comparing)
import System.IO
import System.CPUTime
import Text.Printf

maxDepth :: Int
maxDepth = 600

tolerance :: Double
tolerance = 1e-10

readCoefficients :: FilePath -> IO [Double]
readCoefficients path = do
  content <- readFile path
  return $ map read $ lines content

cauchyBound :: [Double] -> Double
cauchyBound (a:as) = 1 + maximum (map (abs . (/ a)) as)

fujiwaraBound :: [Double] -> Double
fujiwaraBound coeffs@(a:as) =
  let n = length coeffs - 1
      powers = zipWith (\c j -> 2 * (abs (c / a)) ** (1 / fromIntegral (n - j + 1))) as [1..]
  in maximum powers


kojimaBound :: [Double] -> Double
kojimaBound (a:as) = maximum $ map (\c -> 2 * abs (c / a)) as

derivePoly :: [Double] -> [Double]
derivePoly coeffs =
  let n = length coeffs - 1
  in zipWith (*) (map fromIntegral [n, n-1..1]) coeffs

polyVal :: [Double] -> Double -> Double
polyVal coeffs x =
  sum $ zipWith (*) coeffs (reverse $ map (x **) [0..fromIntegral (length coeffs - 1)])

polyValSign :: [Double] -> Double -> Double
polyValSign p x
  | abs x <= 1 = signum $ polyVal p x
  | otherwise =
      let y = 1 / x
          n = length p - 1
          revp = reverse p
      in if even n then signum $ polyVal revp y else signum $ polyVal revp y * signum y

polyFraction :: [Double] -> [Double] -> Double -> Double
polyFraction p q x =
  let n = length p - length q
  in if abs x > 1
     then let lead = x ^^ n
              y = 1 / x
              pr = reverse p
              qr = reverse q
          in lead * polyVal pr y / polyVal qr y
     else polyVal p x / polyVal q x

bisection :: [Double] -> Double -> Double -> Double -> Maybe Double
bisection coeffs a b tol =
  let signA = polyValSign coeffs a
      signB = polyValSign coeffs b
  in if signA * signB > 0 then Nothing
     else go a b
  where
    go a b
      | b - a < tol = Just $ (a + b) / 2
      | otherwise =
          let m = (a + b) / 2
              sm = polyValSign coeffs m
          in if sm == 0 then Just m
             else if polyValSign coeffs a * sm < 0 then go a m else go m b

newtonRaphson :: [Double] -> [Double] -> Double -> Double -> Double -> Double -> Int -> Maybe Double
newtonRaphson p dp x0 a b tol 0 = Nothing
newtonRaphson p dp x0 a b tol n =
  let ratio = polyFraction p dp x0
      x1 = x0 - ratio
  in if abs (x1 - x0) < tol then Just x1
     else if x1 < a || x1 > b then Nothing
     else newtonRaphson p dp x1 a b tol (n - 1)

findRoots :: [Double] -> Double -> Double -> Int -> [Double]
findRoots coeffs a b depth =
  let n = length coeffs
  in if depth > maxDepth
       then []
       else if n <= 2
         then [-head coeffs / coeffs !! 1]
         else
           let d_coeffs = derivePoly coeffs
               d_roots = findRoots d_coeffs a b (depth + 1)
               allPoints = sort (a : b : d_roots)
               dp = derivePoly coeffs
           in foldr (\(x1, x2) acc ->
                case bisection coeffs x1 x2 1e-6 of
                  Just m -> case newtonRaphson coeffs dp m x1 x2 tolerance 500 of
                              Just root -> root : acc
                              Nothing -> acc
                  Nothing -> acc
              ) [] (zip allPoints (tail allPoints))

main :: IO ()
main = do
  coeffs <- readCoefficients "poly_coeff_newton.csv"

  start <- getCPUTime
  let bound = fujiwaraBound coeffs
  let roots = findRoots coeffs (-bound) bound 0
  end <- getCPUTime
  putStrLn $ "Newton-Raphson and Bisection method roots: " ++ show roots
  printf "Time taken: %.10f seconds\n" (fromIntegral (end - start) / 1e12 :: Double)
