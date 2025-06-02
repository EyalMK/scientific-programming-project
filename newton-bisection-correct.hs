import Debug.Trace (trace)

import Data.Time.Clock
import Data.List (sort)
import Numeric.LinearAlgebra (Vector, fromList, toList, dot, cmap, maxElement, size)

tolerance :: Double
tolerance = 1e-6

maxItr :: Int
maxItr = 50


readCoefficients :: FilePath -> IO [Double]
readCoefficients path = do
  content <- readFile path
  return $ map read $ lines content

fujiwaraBound :: [Double] -> Double
fujiwaraBound (a_n:coeffs) =
    let n = length coeffs
        ks = [1..fromIntegral n]
        bs = zipWith (\a k -> (abs (a / a_n)) ** (1.0 / k)) coeffs ks
    in 2.0 * maximum bs

normalizeByMaxCoefficient :: Vector Double -> Vector Double
normalizeByMaxCoefficient coeffs =
    let maxCoeff = maxElement (abs coeffs)
    in if maxCoeff /= 0
       then cmap (\c -> c / maxCoeff) coeffs
       else coeffs

polyVal :: Vector Double -> Double -> Double
polyVal coeffs x =
    let coeffList = toList coeffs
        powers = reverse $ map (x **) [0..fromIntegral (length coeffList - 1)]
    in sum $ zipWith (*) coeffList powers

signOf :: Double -> Double -> Double
signOf eps val
  | abs val < eps = 0.0
  | otherwise     = signum val

polyValSign :: Vector Double -> Double -> Double
polyValSign p x =
    if abs x <= 1
    then signOf tolerance $ polyVal p x
    else let n = size p - 1
             y = 1 / x
             reversedP = fromList . reverse . toList $ p
             in if even n
             then signOf tolerance $ polyVal reversedP y
             else (signOf tolerance $ polyVal reversedP y) * (signOf tolerance y)

polyFraction :: Vector Double -> Vector Double -> Double -> Double
polyFraction p q x =
    let n = size p - size q
    in if abs x > 1
    then
        let pCoefficients = fromList . reverse . toList $ p
            qCoefficients = fromList . reverse . toList $ q
            y = 1 / x
            leadCoefficient = x ** fromIntegral n
            in leadCoefficient * polyVal pCoefficients y / polyVal qCoefficients y
    else polyVal p x / polyVal q x

polyDerivative :: Vector Double -> Vector Double
polyDerivative p =
    let n = size p - 1
        coeffs = toList p
        derivedCoeffs = zipWith (*) (map fromIntegral [n, n-1..1]) coeffs
    in fromList derivedCoeffs


newtonRaphson :: Vector Double -> Vector Double -> Double -> Double -> Double -> Maybe Double
newtonRaphson p dp x0 a b = go x0 0
    where
       go x i
         | i >= maxItr = Nothing
         | not (a <= xNew && xNew <= b) = Nothing
         | abs(xNew - x) < tolerance = Just xNew
         | otherwise = go xNew(i + 1)
         where
            ratio = polyFraction p dp x
            xNew = x - ratio

bisection :: Vector Double -> Double -> Double -> (Double, Bool)
bisection p a b =
    let aSign = polyValSign p a
        bSign = polyValSign p b
    in if aSign == 0.0 then (a, True)
       else if bSign == 0.0 then (b, True)
       else go a b
  where
    go a b
      | b- a <= tolerance = (midpoint, False)
      | midSign == 0.0 || abs (b - a) < tolerance = (midpoint, False)
      | aSign * midSign < 0 = go a midpoint
      | otherwise = go midpoint b
      where
        midpoint = (a + b) / 2
        midSign = polyValSign p midpoint
        aSign = polyValSign p a


newtonRaphsonAndBisectionMethod :: Vector Double -> Double -> Double -> Maybe Double
newtonRaphsonAndBisectionMethod p a b =
    let dp = polyDerivative p
        initialGuess = (a + b) / 2
        newtonRaphsonRoot = newtonRaphson p dp initialGuess a b
    in case newtonRaphsonRoot of
        Just root -> Just root
        Nothing -> let (point, pointIsRoot) = bisection p a b
                    in if pointIsRoot then Just point
                       else newtonRaphson p dp point a b


findRootsInIntervals :: Vector Double -> Vector Double -> Vector Double
findRootsInIntervals p intervals = fromList (go (toList intervals))
  where
    go [] = []
    go [_] = []
    go (x1:x2:xs)
      | polyValSign p x1 * polyValSign p x2 < 0 =
          case newtonRaphsonAndBisectionMethod p x1 x2 of
            Just root -> root : go (x2:xs)
            Nothing -> go (x2:xs)
      | otherwise = go (x2:xs)

findRoots :: Vector Double -> Double -> Double -> Vector Double
findRoots p a b
  | size p <= 2 =
      let pList = toList p
      in fromList [ - (pList !! 1) / (pList !! 0) ]
  | otherwise =
      let criticalPoints = toList $ findRoots (normalizeByMaxCoefficient (polyDerivative p)) a b
          endpoints = sort (a : b : criticalPoints)
      in findRootsInIntervals p (fromList endpoints)

main :: IO ()
main = do
  p <- readCoefficients "poly_coeff_newton.csv"
  let pVector = fromList p

  start <- getCurrentTime
  let bound = fujiwaraBound p
  let roots = findRoots pVector (-bound) bound
  end <- getCurrentTime

  putStrLn $ "Newton-Raphson & Bisection real roots: " ++ show (toList roots)
  print $ diffUTCTime end start