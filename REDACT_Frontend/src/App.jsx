import React, { useState } from "react";
import ButtonGradient from "./assets/svg/ButtonGradient";
import Allroutes from "./components/Allroutes";
import Loader from "./components/Loader";

const App = () => {
  const [isLoading, setIsLoading] = useState(true);

  const handleLoadComplete = () => {
    setIsLoading(false); // Hide loader after loading
  };
 
    // Check screen size and update state
    

  return (
    <>
      {isLoading ? (
        <Loader onLoadComplete={handleLoadComplete} />
      ) : (
        <>
          <Allroutes />
          <ButtonGradient />
        </>
      )}
    </>
  );
};

export default App;
