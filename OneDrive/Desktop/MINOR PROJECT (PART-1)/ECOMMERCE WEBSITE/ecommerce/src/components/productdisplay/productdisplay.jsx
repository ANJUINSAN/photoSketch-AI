import React, { useContext } from "react";
import "./productdisplay.css";
import star_icon from '../images/Assets/star_icon.png'
import star_dull_icon from '../images/Assets/star_dull_icon.png'
import { ShopContext } from "../../context/shopcontext";

export default function productdisplay(props) {
  const { product } = props;
  const {all_product,addToCart}=useContext(ShopContext);

  return (
    <div className="productdisplay">
      <div className="productdisplay-left">
        <div className="productdisplay-img-list">
          <img className="c1" src={product.image} alt="" />
          <img className="c2" src={product.image} alt="" />
          <img className="c3" src={product.image} alt="" />
          <img className="c4" src={product.image} alt="" />
        </div>
        <div className="productdisplay-img">
          <img className="productdisplay-main-img" src={product.image} alt="" />
        </div>
      </div>
      <div className="productdisplay-right">
        <h1>{product.name}</h1>
        <div className="productdisplay-right-stars">
          <img src={star_icon} alt="" />
          <img src={star_icon} alt="" />
          <img src={star_icon} alt="" />
          <img src={star_icon} alt="" />
          <img src={star_dull_icon} alt="" />
          <p>{122}</p>
        </div>
        <div className="productdisplay-right-prices">
            <div className="productdisplay-right-prices-old">${product.old_price}</div>
            <div className="productdisplay-right-prices-new">${product.new_price}</div>
        </div>
        <div className="productdisplay-right-description">
            A lightweight,usually knitted,close fitting,short-sleevesand round neckline
        </div>
        <div className="productdisplay-right-size">
            <h1>Select size</h1>
            <div className="productdisplay-right-sizes">
                <div>S</div>
                <div>M</div>
                <div>L</div>
                <div>XL</div>
                <div>FREE SIZE</div>
            </div>
        </div>
        <button onClick={()=>{addToCart(product.id)}}>ADD TO CART</button>
        <p className="productdisplay-right-category">
            <span>Category:</span> {product.category},t-shirt,shirt,inner
        </p>
        <p className="productdisplay-right-category">
            <span>Tags:</span>modern,latest
        </p>

      </div>
    </div>
  );
}
