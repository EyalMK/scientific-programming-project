import System.IO (openFile, hPutStrLn, IOMode(WriteMode), hClose)
import Data.Time.Clock
import Data.List (sort, nub)
import Numeric.LinearAlgebra (Vector, fromList, toList, dot, cmap, maxElement, size)

tolerance :: Double
tolerance = 1e-8

maxItr :: Int
maxItr = 50

maxDepth :: Int
maxDepth = 10000


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

normalizeByMaxCoefficient :: [Double] -> [Double]
normalizeByMaxCoefficient coeffs =
  let maxCoeff = maximum (map abs coeffs)
  in if maxCoeff /= 0 then map (/ maxCoeff) coeffs else coeffs

polyVal :: [Double] -> Double -> Double
polyVal coeffs x =
  sum $ zipWith (*) coeffs (reverse $ map (x **) [0..fromIntegral (length coeffs - 1)])

signOf :: Double -> Double -> Double
signOf eps val
  | abs val < eps = 0.0
  | otherwise     = signum val

polyValSign :: [Double] -> Double -> Double
polyValSign p x =
    if abs x <= 1
    then signOf tolerance $ polyVal p x
    else let n = length p - 1
             y = 1 / x
             reversedP = reverse p
             in if even n
             then signOf tolerance $ polyVal reversedP y
             else (signOf tolerance $ polyVal reversedP y) * (signOf tolerance y)

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

polyDerivative :: [Double] -> [Double]
polyDerivative p =
    let n = length p - 1
    in zipWith (*) (map fromIntegral [n, n-1..1]) p

newtonRaphson :: [Double] -> [Double] -> Double -> Double -> Double -> Int -> Maybe Double
newtonRaphson p dp x0 a b 0 = Nothing
newtonRaphson p dp x0 a b iteration =
  let ratio = polyFraction p dp x0
      x1 = x0 - ratio
  in if abs (x1 - x0) < tolerance then Just x1
     else if not (a <= x1 && x1 <= b) then Nothing
     else newtonRaphson p dp x1 a b (iteration - 1)

bisection :: [Double] -> Double -> Double -> (Double, Bool)
bisection p a b =
    let aSign = polyValSign p a
        bSign = polyValSign p b
    in if aSign == 0.0
          then (a, True)
       else if bSign == 0.0
          then (b, True)
       else go a b
  where
    go a b
      | b - a < tolerance = ((a + b) / 2, False)
      | otherwise =
          let midpoint = (a + b) / 2
              midSign = polyValSign p midpoint
          in if midSign == 0.0 || abs (b - a) < tolerance
                then (midpoint, False)
             else if polyValSign p a * midSign < 0
                then go a midpoint
                else go midpoint b

findRoots :: [Double] -> Double -> Double -> Int -> [Double]
findRoots coeffs a b depth
  | depth > maxDepth = []
  | degree == 0 = []
  | degree == 1 =
      let [a1, a0] = coeffs
      in if a1 == 0 then [] else
         let root = -a0 / a1
         in [root | a <= root, root <= b]
  | otherwise =
       let d_coeffs = polyDerivative coeffs
           d_roots = if all (== 0) d_coeffs then []
                     else nub $ findRoots d_coeffs a b (depth + 1)
           allPoints = sort (a : b : d_roots)
           checkInterval x1 x2 acc =
             let p_low  = polyValSign coeffs x1
                 p_high = polyValSign coeffs x2
             in if p_low == 0.0
                   then x1 : acc
                else if p_high == 0.0
                   then x2 : acc
                else if p_low * p_high < 0
                   then
                     let (point, isRoot) = bisection coeffs x1 x2
                     in if isRoot
                           then point : acc
                        else
                           case newtonRaphson coeffs dp point x1 x2 maxItr of
                             Just root -> root : acc
                             Nothing   -> acc
                else acc
       in nub $ foldr (\(x1, x2) acc -> checkInterval x1 x2 acc)
                [] (zip allPoints (tail allPoints))
  where
    degree = length coeffs - 1
    dp = polyDerivative coeffs

main :: IO ()
main = do
  p <- readCoefficients "poly_coeff_newton.csv"
  let bound = fujiwaraBound p

  let dp = polyDerivative p
  let dproots = sort $ findRoots (normalizeByMaxCoefficient dp) (-bound) bound 0
  let endpoints = sort ((-bound) : dproots ++ [bound])
  let values = [polyVal p x | x <- endpoints]
  let signs = [polyValSign p x | x <- endpoints]
  let intervals = zip endpoints (tail endpoints)
  let intervalSigns = [(a, b, polyValSign p a, polyValSign p b) | (a, b) <- intervals]
  let rootIntervals = [(a, b) | (a, b, s1, s2) <- intervalSigns, s1 * s2 < 0]

  -- Open file for writing
  h <- openFile "haskell_output.txt" WriteMode
  hPutStrLn h $ "Coefficients: " ++ show p
  hPutStrLn h $ "Fujiwara bound: " ++ show bound
  hPutStrLn h $ "Derivative roots (dproots): " ++ show dproots
  hPutStrLn h $ "Endpoints: " ++ show endpoints
  hPutStrLn h $ "Values at endpoints: " ++ show values
  hPutStrLn h $ "Signs at endpoints: " ++ show signs
  hPutStrLn h $ "Intervals and their endpoint signs: " ++ show intervalSigns
  hPutStrLn h $ "Intervals chosen for root search (sign change): " ++ show rootIntervals

  start <- getCurrentTime
  let roots = findRoots p (-bound) bound 0
  end <- getCurrentTime

  putStrLn $ "Newton-Raphson & Bisection real roots: " ++ show roots
  print $ diffUTCTime end start

  hPutStrLn h $ "Roots found: " ++ show roots
  hPutStrLn h $ "Time taken: " ++ show (realToFrac (diffUTCTime end start) :: Double) ++ " seconds"
  hClose h